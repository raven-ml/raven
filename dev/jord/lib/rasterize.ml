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

(* τ, the tolerance of [touched], in pixels. *)
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
}

let scratch () =
  {
    edges = [||];
    ranges = [||];
    crossings = [||];
    starts = [||];
    mask = Bytes.empty;
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

let touched g p =
  check "touched" g p;
  let pts = pixels g p and s = scratch () and o = out () in
  for f = 0 to Polygons.count p - 1 do
    let first = p.rings.(p.features.(f))
    and last = p.rings.(p.features.(f + 1)) in
    if first < last then begin
      (* The feature's box of cells, within the grid. *)
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
        done;
        for i = r0 to r1 do
          for j = c0 to c1 do
            if Bytes.unsafe_get mask (((i - r0) * w) + (j - c0)) = '\001' then
              emit o f ((i * g.cols) + j)
          done
        done
      end
    end
  done;
  tensors o
