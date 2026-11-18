(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Bigarray

(* Both rules walk each feature alone, in pixel points, on the host.

   centres: a scanline at each row of centres, r = i + 0.5, crosses the edges
   whose row span holds it half-open, a.r <= r < b.r for endpoints a, b in order
   of row. Between the 1st and 2nd crossing, the 3rd and 4th, ... (even-odd), a
   centre j + 0.5 is covered when x₁ < j + 0.5 <= x₂, which is the centre moved
   left by δ and down by ε << δ.

   touched: the centres' cells, and, for each edge and each row it spans, the
   columns of the part of the edge inside the row's open strip (i + τ, i + 1 -
   τ) that meet the open column (j + τ, j + 1 - τ).

   Pairs go out in order of feature, then cell. *)

(* τ, the tolerance of [touched], in pixels. It lies above the error a vertex
   takes from a float64 round trip through another system, a few ulps of a
   coordinate in metres (an ulp of 10⁷ m is under 2e-9 m), on grids of a
   centimetre and coarser, so a polygon traced along cell borders still
   covers only its own cells after that trip, which a GeoJSON file's polygons
   take through CRS84. It lies far below any geometry, 10 µm on a 10 m grid.
   It does not cover coarser errors, such as those of coordinates rounded to
   six decimals of a degree. *)
let tolerance = 1e-6

(* Growable int64 columns of the relation. *)
type out = {
  mutable features : (int64, int64_elt, c_layout) Array1.t;
  mutable cells : (int64, int64_elt, c_layout) Array1.t;
  mutable len : int;
}

let column n = Array1.create int64 c_layout n
let out () = { features = column 1024; cells = column 1024; len = 0 }

let grow o =
  let n = 2 * Array1.dim o.features in
  let copy a =
    let b = column n in
    Array1.blit a (Array1.sub b 0 (Array1.dim a));
    b
  in
  o.features <- copy o.features;
  o.cells <- copy o.cells

let emit o feature cell =
  if o.len = Array1.dim o.features then grow o;
  Array1.unsafe_set o.features o.len (Int64.of_int feature);
  Array1.unsafe_set o.cells o.len (Int64.of_int cell);
  o.len <- o.len + 1

let tensors o =
  let take a = Nx.of_bigarray (genarray_of_array1 (Array1.sub a 0 o.len)) in
  (take o.features, take o.cells)

(* Inlined, as the float helpers below, so the hot loops box no float. *)
let[@inline] fmin (a : float) b = if a < b then a else b
let[@inline] fmax (a : float) b = if a > b then a else b
let[@inline] clamp lo hi x = fmin hi (fmax lo x)

(* Rows and columns by exact comparison with i + 0.5, an exact float for any
   index; the float guess is off by at most one. The caller clamps [x], so the
   index fits. *)

(* The least i with i + 0.5 >= x. *)
let[@inline] first_ge x =
  let i = ref (int_of_float (Float.ceil (x -. 0.5))) in
  while float_of_int !i +. 0.5 < x do
    incr i
  done;
  while float_of_int (!i - 1) +. 0.5 >= x do
    decr i
  done;
  !i

(* The least i with i + 0.5 > x. *)
let[@inline] first_gt x =
  let i = ref (int_of_float (Float.floor (x -. 0.5)) + 1) in
  while float_of_int !i +. 0.5 <= x do
    incr i
  done;
  while float_of_int (!i - 1) +. 0.5 > x do
    decr i
  done;
  !i

(* Scratch space reused across features: edges as (a.r, a.c, b.r, b.c) with a
   above b, their rows, the crossings of each row and where each row's start,
   and the touched mask. *)
type scratch = {
  mutable edges : float array;
  mutable ranges : int array;
  mutable crossings : float array;
  mutable starts : int array;
  mutable mask : Bytes.t;
  (* The box [cover] fills, kept here so that no tuple is allocated per
     feature. *)
  mutable r0 : int;
  mutable r1 : int;
  mutable c0 : int;
  mutable c1 : int;
}

let scratch () =
  {
    edges = [||];
    ranges = [||];
    crossings = [||];
    starts = [||];
    mask = Bytes.empty;
    r0 = 0;
    r1 = -1;
    c0 = 0;
    c1 = -1;
  }

let ensure_float a n = if Array.length a >= n then a else Array.make (2 * n) 0.
let ensure_int a n = if Array.length a >= n then a else Array.make (2 * n) 0

(* The feature's edges that are not horizontal, ordered by row. [pts] holds
   pixel points (r, c) interleaved. Returns their count. *)
let gather s (p : Polygons.t) pts f =
  let first = p.features.(f) and last = p.features.(f + 1) in
  let points = p.rings.(last) - p.rings.(first) in
  s.edges <- ensure_float s.edges (4 * points);
  let e = s.edges in
  let n = ref 0 in
  for k = first to last - 1 do
    let lo = p.rings.(k) and hi = p.rings.(k + 1) in
    for m = lo to hi - 1 do
      let q = if m + 1 = hi then lo else m + 1 in
      let r0 = pts.(2 * m) and c0 = pts.((2 * m) + 1) in
      let r1 = pts.(2 * q) and c1 = pts.((2 * q) + 1) in
      if r0 <> r1 then begin
        let b = 4 * !n in
        if r0 < r1 then (
          e.(b) <- r0;
          e.(b + 1) <- c0;
          e.(b + 2) <- r1;
          e.(b + 3) <- c1)
        else (
          e.(b) <- r1;
          e.(b + 1) <- c1;
          e.(b + 2) <- r0;
          e.(b + 3) <- c0);
        incr n
      end
    done
  done;
  !n

(* Calls [span i j0 j1] for each run of covered centres j0..j1 of row i, in
   order of row, then column. *)
let centre_spans s (g : Grid.t) edges span =
  let e = s.edges in
  let rows = float_of_int g.rows and cols = float_of_int g.cols in
  (* Each edge's rows of centres, i0 <= i < i1, within the grid. *)
  s.ranges <- ensure_int s.ranges (2 * edges);
  let rg = s.ranges in
  let lo = ref g.rows and hi = ref 0 in
  for k = 0 to edges - 1 do
    let i0 = max 0 (first_ge (clamp (-1.) (rows +. 1.) e.(4 * k))) in
    let i1 = min g.rows (first_ge (clamp (-1.) (rows +. 1.) e.((4 * k) + 2))) in
    rg.(2 * k) <- i0;
    rg.((2 * k) + 1) <- i1;
    if i0 < i1 then (
      lo := min !lo i0;
      hi := max !hi i1)
  done;
  if !lo < !hi then begin
    let lo = !lo and hi = !hi in
    let nrows = hi - lo in
    (* Count each row's crossings, then place them: a counting sort. *)
    s.starts <- ensure_int s.starts ((2 * nrows) + 1);
    let st = s.starts in
    Array.fill st 0 ((2 * nrows) + 1) 0;
    for k = 0 to edges - 1 do
      for i = rg.(2 * k) to rg.((2 * k) + 1) - 1 do
        st.(i - lo + 1) <- st.(i - lo + 1) + 1
      done
    done;
    for i = 1 to nrows do
      st.(i) <- st.(i) + st.(i - 1)
    done;
    (* st.(nrows + 1 + r) is the next free slot of row lo + r. *)
    Array.blit st 0 st (nrows + 1) nrows;
    s.crossings <- ensure_float s.crossings st.(nrows);
    let xs = s.crossings in
    for k = 0 to edges - 1 do
      let ar = e.(4 * k) and ac = e.((4 * k) + 1) in
      let br = e.((4 * k) + 2) and bc = e.((4 * k) + 3) in
      for i = rg.(2 * k) to rg.((2 * k) + 1) - 1 do
        let y = float_of_int i +. 0.5 in
        let slot = nrows + 1 + i - lo in
        xs.(st.(slot)) <- ac +. ((y -. ar) *. (bc -. ac) /. (br -. ar));
        st.(slot) <- st.(slot) + 1
      done
    done;
    for i = lo to hi - 1 do
      let a = st.(i - lo) and b = st.(i - lo + 1) in
      (* Insertion sort: a row holds a handful of crossings. *)
      for m = a + 1 to b - 1 do
        let x = xs.(m) in
        let k = ref (m - 1) in
        while !k >= a && xs.(!k) > x do
          xs.(!k + 1) <- xs.(!k);
          decr k
        done;
        xs.(!k + 1) <- x
      done;
      let m = ref a in
      while !m + 1 < b do
        let j0 = max 0 (first_gt (clamp (-1.) (cols +. 1.) xs.(!m))) in
        let j1 = min g.cols (first_gt (clamp (-1.) (cols +. 1.) xs.(!m + 1))) in
        if j0 < j1 then span i j0 (j1 - 1);
        m := !m + 2
      done
    done
  end

let check fn (g : Grid.t) (p : Polygons.t) =
  if not (Crs.equal g.crs p.crs) then
    invalid_arg
      (Format.asprintf
         "Jord.Rasterize.%s: the features are in %a, the grid in %a" fn Crs.pp
         p.crs Crs.pp g.crs)

let pixels (g : Grid.t) (p : Polygons.t) = Nx.to_array (Grid.pixel g p.points)

let centres g p =
  check "centres" g p;
  let pts = pixels g p and s = scratch () and o = out () in
  for f = 0 to Polygons.count p - 1 do
    let edges = gather s p pts f in
    centre_spans s g edges (fun i j0 j1 ->
        let base = i * g.cols in
        for j = j0 to j1 do
          emit o f (base + j)
        done)
  done;
  tensors o

(* Marks in the mask, of the box rows r0..r1 and columns c0..c1, the cells whose
   open square the edge (pr, pc)-(qr, qc) meets. *)
let[@inline] mark_edge mask ~r0 ~r1 ~c0 ~c1 pr pc qr qc =
  let w = c1 - c0 + 1 in
  let dr = qr -. pr and dc = qc -. pc in
  let r0f = float_of_int (r0 - 1) and r1f = float_of_int (r1 + 1) in
  let c0f = float_of_int (c0 - 1) and c1f = float_of_int (c1 + 1) in
  let lo_row =
    max r0 (int_of_float (Float.floor (clamp r0f r1f (fmin pr qr))))
  in
  let hi_row =
    min r1 (int_of_float (Float.floor (clamp r0f r1f (fmax pr qr))))
  in
  for i = lo_row to hi_row do
    let lo = float_of_int i +. tolerance
    and hi = float_of_int (i + 1) -. tolerance in
    (* The part of the edge inside the open strip lo < r < hi, where there is
       one, by its parameter t in [t0, t1] of [0, 1]. *)
    let ta = (lo -. pr) /. dr and tb = (hi -. pr) /. dr in
    let inside =
      if dr = 0. then lo < pr && pr < hi else fmin ta tb < 1. && fmax ta tb > 0.
    in
    if inside then begin
      let t0 = if dr = 0. then 0. else fmax 0. (fmin ta tb) in
      let t1 = if dr = 0. then 1. else fmin 1. (fmax ta tb) in
      (* Fused on purpose: OCaml's arm64 backend fuses [pc +. t *. dc] and
         amd64's does not, and a cell at τ's edge must not depend on which. *)
      let ca = Float.fma t0 dc pc and cb = Float.fma t1 dc pc in
      let u = fmin ca cb and v = fmax ca cb in
      let j0 = max c0 (int_of_float (Float.floor (clamp c0f c1f u))) in
      let j1 = min c1 (int_of_float (Float.floor (clamp c0f c1f v))) in
      for j = j0 to j1 do
        if
          u < float_of_int (j + 1) -. tolerance
          && v > float_of_int j +. tolerance
        then Bytes.unsafe_set mask (((i - r0) * w) + (j - c0)) '\001'
      done
    end
  done

(* Fills the mask with touched's cells of feature [f] over its box of cells
   within the grid, and sets the box, rows s.r0..s.r1 and columns s.c0..s.c1,
   empty where r0 > r1 or c0 > c1. *)
let cover s (g : Grid.t) (p : Polygons.t) pts f =
  let first = p.rings.(p.features.(f))
  and last = p.rings.(p.features.(f + 1)) in
  if first >= last then begin
    s.r0 <- 0;
    s.r1 <- -1;
    s.c0 <- 0;
    s.c1 <- -1
  end
  else begin
    let rmin = ref Float.infinity and rmax = ref Float.neg_infinity in
    let cmin = ref Float.infinity and cmax = ref Float.neg_infinity in
    for m = first to last - 1 do
      let r = pts.(2 * m) and c = pts.((2 * m) + 1) in
      rmin := fmin !rmin r;
      rmax := fmax !rmax r;
      cmin := fmin !cmin c;
      cmax := fmax !cmax c
    done;
    let cell lo hi x = int_of_float (Float.floor (clamp lo hi x)) in
    let rows = float_of_int g.rows and cols = float_of_int g.cols in
    let r0 = max 0 (cell (-1.) rows !rmin)
    and r1 = min (g.rows - 1) (cell (-1.) rows !rmax) in
    let c0 = max 0 (cell (-1.) cols !cmin)
    and c1 = min (g.cols - 1) (cell (-1.) cols !cmax) in
    if r0 <= r1 && c0 <= c1 then begin
      let w = c1 - c0 + 1 in
      let size = (r1 - r0 + 1) * w in
      if Bytes.length s.mask < size then s.mask <- Bytes.create (2 * size);
      let mask = s.mask in
      Bytes.fill mask 0 size '\000';
      let edges = gather s p pts f in
      centre_spans s g edges (fun i j0 j1 ->
          Bytes.fill mask (((i - r0) * w) + (j0 - c0)) (j1 - j0 + 1) '\001');
      for k = p.features.(f) to p.features.(f + 1) - 1 do
        let lo = p.rings.(k) and hi = p.rings.(k + 1) in
        for m = lo to hi - 1 do
          let q = if m + 1 = hi then lo else m + 1 in
          mark_edge mask ~r0 ~r1 ~c0 ~c1
            pts.(2 * m)
            pts.((2 * m) + 1)
            pts.(2 * q)
            pts.((2 * q) + 1)
        done
      done
    end;
    s.r0 <- r0;
    s.r1 <- r1;
    s.c0 <- c0;
    s.c1 <- c1
  end

let touched g p =
  check "touched" g p;
  let pts = pixels g p and s = scratch () and o = out () in
  for f = 0 to Polygons.count p - 1 do
    cover s g p pts f;
    let r0 = s.r0 and r1 = s.r1 and c0 = s.c0 and c1 = s.c1 in
    let w = c1 - c0 + 1 in
    for i = r0 to r1 do
      for j = c0 to c1 do
        if Bytes.unsafe_get s.mask (((i - r0) * w) + (j - c0)) = '\001' then
          emit o f ((i * g.cols) + j)
      done
    done
  done;
  tensors o

(* areas: for each column of a feature's box, the slabs of the edges over it
   (Slab, with columns relative to the column's left side and rows to R, the
   feature's least vertex row rounded down), each edge's windows with their
   signs, then each cell of touched's relation that an edge meets: the sum
   over the column's windows of the sign times hi - lo for those wholly past
   the cell's row, and of the sign times Overlap's term, whose width runs
   along the edge, times the edge's direction along the columns, for those
   that meet it. A cell no edge meets is 1.
   The terms are evaluated in batches, as host tensors.

   Bound. For cell (i, j), let K be the windows summed, W the sum of the
   widths of all the column's windows, M ≥ 1 the largest magnitude of the
   coordinates, from the cell's corner, of the edges whose windows meet the
   row, H ≥ 1 the feature's largest vertex row less R, and u = 2⁻⁵³. To
   first order the area errs by at most (24M + K + 75H) u W + 10 u X, where X
   counts the crossings computed in the column:

   - Terms and sum. Each term errs by (15M + 5) u |w| (Overlap), a window
     wholly past the row adds its sign times hi - lo, within u |w| of its
     exact width, and summing K numbers, each of magnitude at most |w|, adds
     (K - 1) u W. Whether a window meets the row is decided on its heights at
     its ends from the cell's corner, by the term's formula, within 12 u M of
     exact: a window wrongly taken as wholly before or past the row, or a cell
     wrongly taken as met by none, moves the area by at most 12 u M |w|,
     inside the term's own bound. (15M + 4 + K) u W ≤ (24M + K) u W.

   - Ranks. Let δ bound the error of a height Slab computes at a cut. Rows
     less R are exact or within u H, columns less j are exact by Sterbenz
     within the column and otherwise within u |x| of an end whose edge spans
     at least |x| / 4 (so 4 u D of the height, D ≤ H the edge's rise), the
     parameter t within 3 u, then t (yq - yp) and the sum: δ ≤ u (2H + 9D) ≤
     11 u H. A key, the sum of two heights, is within 24 u H, and a height
     difference within δ' = 23 u H. Computed ranks are true ranks but where
     a pair is inverted; by the sorting lemma (if every inverted pair at x
     differs by at most Δ, each rank's height is within Δ of the true one at
     x) and the clip being 1-Lipschitz, the area moves by at most Δ times the
     width of each misranked window. Within a slab:
     (i) a pair that does not cross in it, inverted, has keys within 48 u H,
         so its two heights, whose difference keeps its sign, are within
         48 u H over the slab;
     (ii) a crossing missed because a difference at an end of the pair's
         common span is within δ' of 0 leaves that pair within δ' + 48 u H =
         71 u H on its inverted side;
     (iii) a crossing found is placed within δ' / |f'| + 4 u |f₀| / |f'| + u
         of the true abscissa, f' the slope of the height difference and f₀
         its value at the span's start, so on the slab holding the true
         crossing the pair is within δ' + 4 u H + 48 u H = 75 u H, plus u |f'|
         over a width of at most 75 u H / |f'| + 5 u. Where |f'| is past
         about 1 / u, as for a near-vertical edge, whose slope has no bound,
         u |f'| times that width is no bound either, but the two clipped
         heights differ by at most 1, so a strip 5 u wide costs at most 5 u
         per window whatever f' is: X's term;
     (iv) a crossing found but placed outside the common span (x0, x1) is
         dropped. The true one then lies within δ' / |f'| + 4 u |f₀| / |f'| +
         u of the span's end, a cut, so the estimate of (iii) covers it, on
         the slab next to that end.
     So Δ ≤ 75 u H, and the ranks add at most 75 u H W.

   Clamping into [0, 1], where the exact area lies, only lowers the error. *)

(* The heights of edge (yp, xp)-(yq, xq), relative to the cell's corner, at the
   window's ends, as Overlap.term computes them: 0 if the window lies wholly
   before the cell's row, 2 if wholly past it, 1 if it meets it. *)
let[@inline] classify yp xp yq xq lo hi =
  let dx = xq -. xp and dy = yq -. yp in
  let ya = yp +. (clamp 0. 1. ((lo -. xp) /. dx) *. dy)
  and yb = yp +. (clamp 0. 1. ((hi -. xp) /. dx) *. dy) in
  if ya <= 0. && yb <= 0. then 0 else if ya >= 1. && yb >= 1. then 2 else 1

(* The terms waiting for a batch: per row the edge from the cell's corner and
   the window, yp, xp, yq, xq, lo and hi, its sign, and the area it adds to. *)
type terms = {
  mutable inputs : float array;
  mutable mu : float array;
  mutable target : int array;
  mutable n : int;
}

let grow_terms t =
  let n = (2 * t.n) + 1024 in
  let grow a k z = Array.append a (Array.make ((k * n) - Array.length a) z) in
  t.inputs <- grow t.inputs 6 0.;
  t.mu <- grow t.mu 1 0.;
  t.target <- grow t.target 1 0

let[@inline] push t yp xp yq xq lo hi mu target =
  if t.n = Array.length t.mu then grow_terms t;
  let b = 6 * t.n in
  t.inputs.(b) <- yp;
  t.inputs.(b + 1) <- xp;
  t.inputs.(b + 2) <- yq;
  t.inputs.(b + 3) <- xq;
  t.inputs.(b + 4) <- lo;
  t.inputs.(b + 5) <- hi;
  t.mu.(t.n) <- mu;
  t.target.(t.n) <- target;
  t.n <- t.n + 1

(* A feature's scratch beyond touched's: its edges (rp, cp, rq, cq), their
   rows less R and columns less j for Slab, the column buckets of its edges,
   the rows the windows of one column may meet, and, per cell of its box,
   whether an edge meets it, the sum so far, and its index in the
   relation. *)
type feature = {
  mutable edges : float array;
  mutable rel : float array;
  mutable global : int array;
  mutable starts : int array;
  mutable bucket : int array;
  mutable rows : int array; (* Each window's first and last rows. *)
  mutable met : Bytes.t;
  mutable rowmet : Bytes.t;
  mutable acc : float array;
  mutable slot : int array;
}

let feature () =
  {
    edges = [||];
    rel = [||];
    global = [||];
    starts = [||];
    bucket = [||];
    rows = [||];
    met = Bytes.empty;
    rowmet = Bytes.empty;
    acc = [||];
    slot = [||];
  }

let ensure_bytes b n = if Bytes.length b >= n then b else Bytes.create (2 * n)

(* Feature [f]'s edges in [ft.edges], in their rings' order and direction,
   and its least vertex row. Returns their count. *)
let edges_of ft (p : Polygons.t) pts f =
  let first = p.features.(f) and last = p.features.(f + 1) in
  let n = p.rings.(last) - p.rings.(first) in
  ft.edges <- ensure_float ft.edges (4 * n);
  let e = ft.edges and k = ref 0 and rmin = ref Float.infinity in
  for ring = first to last - 1 do
    let lo = p.rings.(ring) and hi = p.rings.(ring + 1) in
    for m = lo to hi - 1 do
      let q = if m + 1 = hi then lo else m + 1 in
      e.(4 * !k) <- pts.(2 * m);
      e.((4 * !k) + 1) <- pts.((2 * m) + 1);
      e.((4 * !k) + 2) <- pts.(2 * q);
      e.((4 * !k) + 3) <- pts.((2 * q) + 1);
      rmin := fmin !rmin pts.(2 * m);
      incr k
    done
  done;
  (n, Float.floor !rmin)

(* Buckets the edges by the columns c0..c1 whose open interior they meet,
   vertical ones by the column whose interior holds them: column j's edges
   are [bucket.(starts.(j - c0))] to [bucket.(starts.(j - c0 + 1) - 1)]. *)
let buckets ft n ~c0 ~c1 =
  let w = c1 - c0 + 1 in
  let lo = float_of_int (c0 - 1) and hi = float_of_int (c1 + 2) in
  let range k =
    let cp = ft.edges.((4 * k) + 1) and cq = ft.edges.((4 * k) + 3) in
    let a = fmin cp cq and b = fmax cp cq in
    if a = b then
      let j = Float.floor (clamp lo hi a) in
      if j = a then (1, 0) else (int_of_float j, int_of_float j)
    else
      ( max c0 (int_of_float (Float.floor (clamp lo hi a))),
        min c1 (int_of_float (Float.ceil (clamp lo hi b)) - 1) )
  in
  ft.starts <- ensure_int ft.starts (w + 1);
  let st = ft.starts in
  Array.fill st 0 (w + 1) 0;
  for k = 0 to n - 1 do
    let j0, j1 = range k in
    for j = max c0 j0 to min c1 j1 do
      st.(j - c0 + 1) <- st.(j - c0 + 1) + 1
    done
  done;
  for j = 1 to w do
    st.(j) <- st.(j) + st.(j - 1)
  done;
  ft.bucket <- ensure_int ft.bucket st.(w);
  let next = Array.sub st 0 w in
  for k = 0 to n - 1 do
    let j0, j1 = range k in
    for j = max c0 j0 to min c1 j1 do
      ft.bucket.(next.(j - c0)) <- k;
      next.(j - c0) <- next.(j - c0) + 1
    done
  done

(* Column j of the feature's box: its windows, the rows its edges meet, and
   for each such cell of touched's relation its sum so far and its terms. *)
let column_areas s sl ft t ~rr ~r0 ~r1 ~c0 ~w j =
  let e = ft.edges and fj = float_of_int j and rf = float_of_int rr in
  let b0 = ft.starts.(j - c0) and b1 = ft.starts.(j - c0 + 1) in
  ft.rel <- ensure_float ft.rel (4 * (b1 - b0));
  ft.global <- ensure_int ft.global (b1 - b0);
  let rel = ft.rel and n = ref 0 in
  for b = b0 to b1 - 1 do
    let k = ft.bucket.(b) in
    if e.((4 * k) + 1) <> e.((4 * k) + 3) then begin
      rel.(4 * !n) <- e.((4 * k) + 1) -. fj;
      rel.((4 * !n) + 1) <- e.(4 * k) -. rf;
      rel.((4 * !n) + 2) <- e.((4 * k) + 3) -. fj;
      rel.((4 * !n) + 3) <- e.((4 * k) + 2) -. rf;
      ft.global.(!n) <- k;
      incr n
    end
  done;
  let nw = Slab.windows sl rel !n ~lo:0. ~hi:1. in
  let edges = Slab.edges sl and ends = Slab.ends sl and signs = Slab.signs sl in
  let rowmet = ft.rowmet in
  Bytes.fill rowmet 0 (r1 - r0 + 1) '\000';
  ft.rows <- ensure_int ft.rows (2 * nw);
  let rows = ft.rows in
  let lof = float_of_int (r0 - 2 - rr) and hif = float_of_int (r1 + 2 - rr) in
  for v = 0 to nw - 1 do
    let k = ft.global.(edges.(v)) in
    let lo = ends.(4 * v) and hi = ends.((4 * v) + 1) in
    (* The rows whose cell-relative heights may meet it, rounding included. *)
    let ya = ends.((4 * v) + 2) and yb = ends.((4 * v) + 3) in
    let lo_row = Float.floor (clamp lof hif (fmin ya yb))
    and hi_row = Float.floor (clamp lof hif (fmax ya yb)) in
    let first = rr + int_of_float lo_row - 1
    and last = rr + int_of_float hi_row + 1 in
    rows.(2 * v) <- first;
    rows.((2 * v) + 1) <- last;
    let rp = e.(4 * k) and cp = e.((4 * k) + 1) in
    let rq = e.((4 * k) + 2) and cq = e.((4 * k) + 3) in
    for i = max r0 first to min r1 last do
      let fi = float_of_int i in
      if classify (rp -. fi) (cp -. fj) (rq -. fi) (cq -. fj) lo hi = 1 then
        Bytes.unsafe_set rowmet (i - r0) '\001'
    done
  done;
  for b = b0 to b1 - 1 do
    let k = ft.bucket.(b) in
    if e.((4 * k) + 1) = e.((4 * k) + 3) then begin
      let a = fmin e.(4 * k) e.((4 * k) + 2)
      and z = fmax e.(4 * k) e.((4 * k) + 2) in
      let lof = float_of_int (r0 - 1) and hif = float_of_int (r1 + 1) in
      let first = int_of_float (Float.floor (clamp lof hif a))
      and last = int_of_float (Float.ceil (clamp lof hif z)) - 1 in
      for i = max r0 first to min r1 last do
        Bytes.unsafe_set rowmet (i - r0) '\001'
      done
    end
  done;
  for i = r0 to r1 do
    let cell = ((i - r0) * w) + (j - c0) in
    if
      Bytes.unsafe_get rowmet (i - r0) = '\001'
      && Bytes.unsafe_get s.mask cell = '\001'
    then begin
      Bytes.unsafe_set ft.met cell '\001';
      let fi = float_of_int i in
      let far = ref 0. in
      for v = 0 to nw - 1 do
        let lo = ends.(4 * v) and hi = ends.((4 * v) + 1) in
        let sign = signs.(v) in
        if i < rows.(2 * v) then far := !far +. (sign *. (hi -. lo))
        else if i <= rows.((2 * v) + 1) then begin
          let k = ft.global.(edges.(v)) in
          let yp = e.(4 * k) -. fi and xp = e.((4 * k) + 1) -. fj in
          let yq = e.((4 * k) + 2) -. fi and xq = e.((4 * k) + 3) -. fj in
          match classify yp xp yq xq lo hi with
          | 0 -> ()
          | 2 -> far := !far +. (sign *. (hi -. lo))
          | _ ->
              (* The term's width runs along the edge's direction. *)
              let mu = if xq > xp then sign else -.sign in
              push t yp xp yq xq lo hi mu cell
        end
      done;
      ft.acc.(cell) <- !far
    end
  done

(* The growable float64 column of the areas, beside [out]'s. *)
type area_column = {
  mutable area : (float, float64_elt, c_layout) Array1.t;
  mutable len : int;
}

let[@inline] add a v =
  if a.len = Array1.dim a.area then begin
    let b = Array1.create float64 c_layout (2 * a.len) in
    Array1.blit a.area (Array1.sub b 0 a.len);
    a.area <- b
  end;
  Array1.unsafe_set a.area a.len v;
  a.len <- a.len + 1

let areas g p =
  check "areas" g p;
  let pts = pixels g p and s = scratch () and o = out () in
  let sl = Slab.create () and ft = feature () in
  let t = { inputs = [||]; mu = [||]; target = [||]; n = 0 } in
  let a = { area = Array1.create float64 c_layout 1024; len = 0 } in
  let settled = ref 0 in
  (* The waiting terms, added to their areas, and the areas since the last
     batch clamped into [0, 1]. *)
  let flush () =
    if t.n > 0 then begin
      let n = t.n in
      let col c =
        let b = Array1.create float64 c_layout n in
        for r = 0 to n - 1 do
          Array1.unsafe_set b r t.inputs.((6 * r) + c)
        done;
        Nx.of_bigarray (genarray_of_array1 b)
      in
      let v, _ =
        Overlap.term ~lo:(col 4) ~hi:(col 5) (col 0, col 1) (col 2, col 3)
      in
      let v = array1_of_genarray (Nx.to_bigarray float64 v) in
      for r = 0 to n - 1 do
        let i = t.target.(r) in
        a.area.{i} <- a.area.{i} +. (t.mu.(r) *. v.{r})
      done;
      t.n <- 0
    end;
    for i = !settled to a.len - 1 do
      a.area.{i} <- clamp 0. 1. a.area.{i}
    done;
    settled := a.len
  in
  for f = 0 to Polygons.count p - 1 do
    cover s g p pts f;
    let r0 = s.r0 and r1 = s.r1 and c0 = s.c0 and c1 = s.c1 in
    if r0 <= r1 && c0 <= c1 then begin
      let n, rr = edges_of ft p pts f in
      let rr = int_of_float rr in
      let w = c1 - c0 + 1 in
      let size = (r1 - r0 + 1) * w in
      ft.met <- ensure_bytes ft.met size;
      Bytes.fill ft.met 0 size '\000';
      ft.rowmet <- ensure_bytes ft.rowmet (r1 - r0 + 1);
      ft.acc <- ensure_float ft.acc size;
      ft.slot <- ensure_int ft.slot size;
      buckets ft n ~c0 ~c1;
      let first = t.n in
      for j = c0 to c1 do
        column_areas s sl ft t ~rr ~r0 ~r1 ~c0 ~w j
      done;
      for i = r0 to r1 do
        for j = c0 to c1 do
          let cell = ((i - r0) * w) + (j - c0) in
          if Bytes.unsafe_get s.mask cell = '\001' then begin
            ft.slot.(cell) <- a.len;
            emit o f ((i * g.cols) + j);
            add a
              (if Bytes.unsafe_get ft.met cell = '\001' then ft.acc.(cell)
               else 1.)
          end
        done
      done;
      for r = first to t.n - 1 do
        t.target.(r) <- ft.slot.(t.target.(r))
      done;
      if t.n >= 65536 then flush ()
    end
  done;
  flush ();
  let features, cells = tensors o in
  let area = Nx.of_bigarray (genarray_of_array1 (Array1.sub a.area 0 a.len)) in
  (features, cells, area)
