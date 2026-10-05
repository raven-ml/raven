(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg

(* The sample of row [i] and column [j] is [z.(i * nx + j)]. *)
type t = {
  nx : int;
  ny : int;
  xs : float array;
  ys : float array;
  z : float array;
}

(* Constructing *)

let err fmt = Printf.ksprintf (fun s -> invalid_arg ("Field2.v: " ^ s)) fmt

let is_real (type a b) (dtype : (a, b) Nx.dtype) =
  match dtype with
  | Complex64 | Complex128 | Bool | Bit -> false
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int4
  | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64 | UInt64 ->
      true

let coords name n = function
  | None -> Array.init n Float.of_int
  | Some a ->
      let len = Array.length a in
      if len <> n then err "%s has %d coordinates for %d samples" name len n;
      Array.iteri
        (fun i c ->
          if not (Float.is_finite c) then
            err "%s.(%d) = %g is not finite" name i c)
        a;
      if len >= 2 then begin
        let up = a.(1) > a.(0) in
        for i = 1 to len - 1 do
          if not (if up then a.(i) > a.(i - 1) else a.(i) < a.(i - 1)) then
            err "%s is not strictly monotonic at %d" name i
        done;
        if not (Float.is_finite (a.(len - 1) -. a.(0))) then
          err "%s spans more than the largest float" name
      end;
      Array.copy a

let v ?xs ?ys z =
  if Nx.ndim z <> 2 then err "the tensor has %d dimensions, not 2" (Nx.ndim z);
  if not (is_real (Nx.dtype z)) then err "the tensor is not real";
  let shape = Nx.shape z in
  let ny = shape.(0) and nx = shape.(1) in
  let xs = coords "xs" nx xs and ys = coords "ys" ny ys in
  { nx; ny; xs; ys; z = Nx.to_array (Nx.cast Nx.float64 z) }

(* Tracing

   A band is traced in index space, where the sample of row [i] and column [j]
   lies at [(j, i)] and a cell's corners in positive order are its top left, top
   right, bottom right and bottom left samples. Each sample has a class: 0 below
   [lo], 1 in the band, 2 at or above [hi], 3 missing.

   The band's polygons in a piece are found on the piece's boundary entries, its
   corners and its crossings in positive order: arcs of the boundary in the band
   are joined by the chords of [R lo], followed forwards, and of [R hi],
   followed backwards. Entries that coincide in the plane, such as a crossing at
   a sample equal to its level, are then merged into one point, and a polygon
   whose points all lie on one side of the piece encloses no area and is
   dropped. Points carry a global identity, a sample index or a crossing's, the
   same in the two pieces that share a side.

   Points are thus distinct in the plane unless merged: a crossing on a cell's
   side lies on its line, and one on a triangle's diagonal is an end of it or
   off the lines of the other sides.

   An edge of a polygon between two points of one side runs along it, through
   every point between them, as atomic edges from point to point; any other edge
   is a chord. Two pieces sharing a side cover each atomic edge of it in
   opposite directions, and such twins cancel. The band's rings are the
   remaining edges, chained by turning, at the end of each edge, through the
   twins around that point inside the band: rings that touch at a point are
   split there, never crossed. *)

(* In index space positive orientation is that of the plane unless exactly one
   axis has decreasing coordinates. *)
let mirrored f = f.xs.(1) < f.xs.(0) <> (f.ys.(1) < f.ys.(0))

type ctx = {
  f : t;
  lo : float;
  hi : float;
  cls : Bytes.t;
  visited : int array; (* Per cell, a bit per edge of its piece traced. *)
}

let cls ctx k = Char.code (Bytes.unsafe_get ctx.cls k)

(* The node of corner [q] of the cell whose top left node is [tl]. *)
let cell_node nx tl q =
  match q with 0 -> tl | 1 -> tl + 1 | 2 -> tl + nx + 1 | _ -> tl + nx

let has_piece ctx c =
  let nx = ctx.f.nx in
  let tl = (c / (nx - 1) * nx) + (c mod (nx - 1)) in
  let missing = ref 0 in
  for q = 0 to 3 do
    if cls ctx (cell_node nx tl q) = 3 then incr missing
  done;
  !missing <= 1

let max_entries = 12
let max_edges = 32

(* A piece and the edges of the band's polygons in it. Side [k] runs from corner
   [k] to corner [k + 1]. Entries are in positive order, corner [k] first among
   those of side [k]; points are entries merged, in the same order. *)
type piece = {
  mutable cell : int;
  mutable m : int; (* Corners: 3 or 4, 0 if the cell is left out. *)
  mutable split_lo : bool;
  mutable split_hi : bool;
  node : int array;
  corner_cls : int array;
  sid : int array; (* Side [k]'s global side identity. *)
  nbr : int array; (* The cell across side [k], or [-1] if it has no piece. *)
  mutable nt : int;
  t_lvl : int array;
      (* [-1] for a corner, [0] for a crossing of [lo], [1] of [hi]. *)
  t_in : bool array; (* A crossing walked into the region of its level. *)
  t_b : int array; (* The entry's point. *)
  mutable nb : int;
  b_x : float array;
  b_y : float array;
  b_gid : int array;
  b_rank : int array; (* 0 for a corner, 1 and 2 for crossings of [lo], [hi]. *)
  b_sides : int array; (* Bit [k] set iff the point is on side [k]. *)
  b_seg : int array; (* The side from the point to the next. *)
  poly : int array;
  mutable ne : int;
  e_src : int array;
  e_dst : int array;
  e_side : int array; (* The side the edge runs along, or [-1]. *)
  e_next : int array;
}

let make_piece () =
  let ints n = Array.make n 0 in
  {
    cell = -1;
    m = 0;
    split_lo = false;
    split_hi = false;
    node = ints 4;
    corner_cls = ints 4;
    sid = ints 4;
    nbr = ints 4;
    nt = 0;
    t_lvl = ints max_entries;
    t_in = Array.make max_entries false;
    t_b = ints max_entries;
    nb = 0;
    b_x = Array.make max_entries 0.;
    b_y = Array.make max_entries 0.;
    b_gid = ints max_entries;
    b_rank = ints max_entries;
    b_sides = ints max_entries;
    b_seg = ints max_entries;
    poly = ints max_entries;
    ne = 0;
    e_src = ints max_edges;
    e_dst = ints max_edges;
    e_side = ints max_edges;
    e_next = ints max_edges;
  }

(* The crossing of [l] between an out sample [vo] and an in sample [vi], at [t]
   from the out end, never overflows; [t <= 1.] since [l <= vi]. *)
let[@inline] ratio l vo vi =
  let d = vi -. vo in
  if Float.is_finite d then (l -. vo) /. d
  else ((l *. 0.5) -. (vo *. 0.5)) /. ((vi *. 0.5) -. (vo *. 0.5))

(* A coordinate of the crossing: monotone in [t], on the side, and exactly the
   in end at [t = 1.]. Crossings of two levels on a side thus keep their order
   and a side's constant coordinate stays exact. *)
let[@inline] lerp co ci t =
  if t = 1. then ci
  else
    let c = co +. (t *. (ci -. co)) in
    if co <= ci then Float.min ci (Float.max co c)
    else Float.max ci (Float.min co c)

(* A crossing on a triangle's diagonal lies off it by rounding error. Its
   parameter is rounded to a multiple of this step, a power of two worth at
   least 256 rounding errors of the diagonal's coordinates over its extent, so
   that the crossings of two levels coincide or lie apart and a crossing is an
   end or off the lines of the other sides, which two rounding errors would
   ensure. Between two crossings that lie apart the diagonal's points can turn
   by their rounding error over their distance, and a chord leaving one of them
   at a shallower angle can cross the other level's chord: the larger step makes
   this rare and still moves crossings by a negligible amount unless the
   coordinates are large against the grid's spacing. The step depends on the
   diagonal only, so that isolines and bands place a crossing alike. *)
let diagonal_step xa xb ya yb =
  let need ca cb =
    let d = Float.abs (cb -. ca) in
    256. *. epsilon_float *. (Float.max (Float.abs ca) (Float.abs cb) +. d) /. d
  in
  Float.ldexp 1. (snd (Float.frexp (Float.max (need xa xb) (need ya yb))))

let[@inline] add_point p t x y gid rank sides seg =
  let nb = p.nb in
  if nb > 0 && x = p.b_x.(nb - 1) && y = p.b_y.(nb - 1) then begin
    let b = nb - 1 in
    if rank < p.b_rank.(b) then begin
      p.b_rank.(b) <- rank;
      p.b_gid.(b) <- gid
    end;
    p.b_sides.(b) <- p.b_sides.(b) lor sides;
    p.b_seg.(b) <- seg;
    p.t_b.(t) <- b
  end
  else begin
    p.b_x.(nb) <- x;
    p.b_y.(nb) <- y;
    p.b_gid.(nb) <- gid;
    p.b_rank.(nb) <- rank;
    p.b_sides.(nb) <- sides;
    p.b_seg.(nb) <- seg;
    p.t_b.(t) <- nb;
    p.nb <- nb + 1
  end

(* The entries and their points. A side crosses [lo] iff one end is below it and
   the other not, [hi] iff one end is at or above it and the other not; its
   crossings come in the order of their levels from its lower end. The last
   point is merged into the first, corner 0, if they coincide. *)
let entries ctx p =
  let f = ctx.f in
  let m = p.m and nx = f.nx and n = f.nx * f.ny in
  p.nt <- 0;
  p.nb <- 0;
  for k = 0 to m - 1 do
    let k' = if k = m - 1 then 0 else k + 1 in
    let a = p.node.(k) and b = p.node.(k') in
    let ca = p.corner_cls.(k) and cb = p.corner_cls.(k') in
    let xa = f.xs.(a mod nx) and ya = f.ys.(a / nx) in
    let xb = f.xs.(b mod nx) and yb = f.ys.(b / nx) in
    let t = p.nt in
    p.t_lvl.(t) <- -1;
    add_point p t xa ya a 0
      ((1 lsl k) lor (1 lsl if k = 0 then m - 1 else k - 1))
      k;
    p.nt <- t + 1;
    let step = if m = 3 && k = 2 then diagonal_step xa xb ya yb else 0. in
    let va = f.z.(a) and vb = f.z.(b) in
    let lower = Int.min ca cb and upper = Int.max ca cb in
    for r = 0 to 1 do
      let lvl = if ca < cb then r else 1 - r in
      let crosses =
        if lvl = 0 then lower = 0 && upper > 0 else lower < 2 && upper = 2
      in
      if crosses then begin
        let l = if lvl = 0 then ctx.lo else ctx.hi in
        let inward = va < l in
        let u = if inward then ratio l va vb else ratio l vb va in
        let u = if step = 0. then u else Float.round (u /. step) *. step in
        let x = if inward then lerp xa xb u else lerp xb xa u in
        let y = if inward then lerp ya yb u else lerp yb ya u in
        let t = p.nt in
        p.t_lvl.(t) <- lvl;
        p.t_in.(t) <- inward;
        add_point p t x y (n + (2 * p.sid.(k)) + lvl) (1 + lvl) (1 lsl k) k;
        p.nt <- t + 1
      end
    done
  done;
  let last = p.nb - 1 in
  if last > 0 && p.b_x.(last) = p.b_x.(0) && p.b_y.(last) = p.b_y.(0) then begin
    if p.b_rank.(last) < p.b_rank.(0) then begin
      p.b_rank.(0) <- p.b_rank.(last);
      p.b_gid.(0) <- p.b_gid.(last)
    end;
    p.b_sides.(0) <- p.b_sides.(0) lor p.b_sides.(last);
    for t = 0 to p.nt - 1 do
      if p.t_b.(t) = last then p.t_b.(t) <- 0
    done;
    p.nb <- last
  end

let[@inline] next_point p b = if b = p.nb - 1 then 0 else b + 1

(* No corner lies between points [b], included, and [w]. *)
let rec no_corner_until p b w =
  b = w || (p.b_rank.(b) <> 0 && no_corner_until p (next_point p b) w)

(* The edge from [u] to [w] runs along a side iff no corner lies between. *)
let along_side p u w = no_corner_until p (next_point p u) w

let add_edge p u w side =
  let e = p.ne in
  if e = max_edges then failwith "Field2: too many edges in a piece";
  p.e_src.(e) <- u;
  p.e_dst.(e) <- w;
  p.e_side.(e) <- side;
  p.ne <- e + 1

(* The polygon of the [np] distinct points of [p.poly], dropped if they are
   fewer than three or all on one side. *)
let add_polygon p np =
  let common = ref (-1) in
  for q = 0 to np - 1 do
    common := !common land p.b_sides.(p.poly.(q))
  done;
  if np >= 3 && !common = 0 then begin
    let first = p.ne in
    for q = 0 to np - 1 do
      let u = p.poly.(q) and w = p.poly.(if q = np - 1 then 0 else q + 1) in
      if along_side p u w then begin
        let b = ref u in
        while !b <> w do
          let b' = next_point p !b in
          add_edge p !b b' p.b_seg.(!b);
          b := b'
        done
      end
      else add_edge p u w (-1)
    done;
    for e = first to p.ne - 2 do
      p.e_next.(e) <- e + 1
    done;
    p.e_next.(p.ne - 1) <- first
  end

(* The next entry from [t] in direction [step] that crosses level [lvl] inwards
   iff [inward]. *)
let rec find p t step lvl inward =
  let k = (t + step + p.nt) mod p.nt in
  if p.t_lvl.(k) = lvl && p.t_in.(k) = inward then k
  else find p k step lvl inward

(* From the end [t] of an arc, the start of the next one. Leaving the region of
   [lo], its chord leads to an entry: the next one if the cell joins its in
   corners or has two crossings, the one that started the run otherwise.
   Entering the region of [hi], its chord is followed backwards to an exit. *)
let jump p t =
  if p.t_lvl.(t) = 0 then
    if p.split_lo then find p t (-1) 0 true else find p t 1 0 true
  else if p.split_hi then find p t 1 1 false
  else find p t (-1) 1 false

let is_arc_start p t =
  match p.t_lvl.(t) with 0 -> p.t_in.(t) | 1 -> not p.t_in.(t) | _ -> false

(* [np] points of the polygon being built, with entry [t]'s point after them
   unless it is the last one already. *)
let[@inline] add_entry p np t =
  let b = p.t_b.(t) in
  if np > 0 && p.poly.(np - 1) = b then np
  else begin
    p.poly.(np) <- b;
    np + 1
  end

let polygons p =
  let nt = p.nt in
  let used = ref 0 in
  for s = 0 to nt - 1 do
    if is_arc_start p s && !used land (1 lsl s) = 0 then begin
      let np = ref 0 and t = ref s and closed = ref false in
      while not !closed do
        used := !used lor (1 lsl !t);
        np := add_entry p !np !t;
        let k = ref ((!t + 1) mod nt) in
        while p.t_lvl.(!k) < 0 do
          np := add_entry p !np !k;
          k := (!k + 1) mod nt
        done;
        np := add_entry p !np !k;
        let j = jump p !k in
        if j = s then closed := true
        else if !used land (1 lsl j) <> 0 then
          failwith "Field2: an arc of a piece is reached twice"
        else t := j
      done;
      if !np > 1 && p.poly.(!np - 1) = p.poly.(0) then decr np;
      add_polygon p !np
    end
  done;
  if nt = p.m && p.corner_cls.(0) = 1 then begin
    for k = 0 to p.m - 1 do
      p.poly.(k) <- p.t_b.(k)
    done;
    add_polygon p p.m
  end

(* Whether a whole cell's in corners at a level, those of class [at_least] or
   more, are two opposite ones that stay apart: their mean is below the
   level. *)
let[@inline] split p at_least l mean =
  let c = p.corner_cls in
  let in0 = c.(0) >= at_least and in1 = c.(1) >= at_least in
  let in2 = c.(2) >= at_least and in3 = c.(3) >= at_least in
  p.m = 4 && in0 = in2 && in1 = in3 && in0 <> in1 && mean < l

(* Side [k] of the piece is side [q] of cell [c]. *)
let set_side ctx p k c q =
  let f = ctx.f in
  let nx = f.nx in
  let ncx = nx - 1 and ncy = f.ny - 1 in
  let i = c / ncx and j = c mod ncx in
  let n_h = f.ny * ncx in
  p.sid.(k) <-
    (match q with
    | 0 -> c
    | 1 -> n_h + (i * nx) + j + 1
    | 2 -> c + ncx
    | _ -> n_h + (i * nx) + j);
  let across =
    match q with
    | 0 -> if i > 0 then c - ncx else -1
    | 1 -> if j + 1 < ncx then c + 1 else -1
    | 2 -> if i + 1 < ncy then c + ncx else -1
    | _ -> if j > 0 then c - 1 else -1
  in
  p.nbr.(k) <- (if across >= 0 && has_piece ctx across then across else -1)

let fill ctx p c =
  if p.cell <> c then begin
    p.cell <- c;
    p.ne <- 0;
    let f = ctx.f in
    let nx = f.nx in
    let ncx = nx - 1 in
    let tl = (c / ncx * nx) + (c mod ncx) in
    let n_missing = ref 0 and missing = ref 0 in
    for q = 0 to 3 do
      if cls ctx (cell_node nx tl q) = 3 then begin
        incr n_missing;
        missing := q
      end
    done;
    if !n_missing > 1 then p.m <- 0
    else begin
      if !n_missing = 0 then begin
        p.m <- 4;
        for k = 0 to 3 do
          p.node.(k) <- cell_node nx tl k;
          set_side ctx p k c k
        done
      end
      else begin
        p.m <- 3;
        let q0 = !missing in
        for k = 0 to 2 do
          p.node.(k) <- cell_node nx tl ((q0 + 1 + k) land 3)
        done;
        set_side ctx p 0 c ((q0 + 1) land 3);
        set_side ctx p 1 c ((q0 + 2) land 3);
        p.sid.(2) <- (f.ny * ncx) + ((f.ny - 1) * nx) + c;
        p.nbr.(2) <- -1
      end;
      for k = 0 to p.m - 1 do
        p.corner_cls.(k) <- cls ctx p.node.(k)
      done;
      let mean =
        if p.m = 4 then
          (f.z.(p.node.(0)) *. 0.25)
          +. (f.z.(p.node.(1)) *. 0.25)
          +. ((f.z.(p.node.(2)) *. 0.25) +. (f.z.(p.node.(3)) *. 0.25))
        else 0.
      in
      p.split_lo <- split p 1 ctx.lo mean;
      p.split_hi <- split p 2 ctx.hi mean;
      entries ctx p;
      polygons p
    end
  end

(* Pieces are kept in slots indexed by their cell modulo a power of two above a
   row of cells, so that a piece and the pieces across its sides never share a
   slot. *)
type tracer = {
  ctx : ctx;
  slots : piece option array;
  mutable n : int;
  mutable rx : float array;
  mutable ry : float array;
  mutable rd : Bytes.t; (* Edge [i] is on the domain's boundary. *)
}

(* The piece of cell [c]. It stays valid until a piece is read for a cell that
   is neither [c] nor across a side of it. *)
let piece tr c =
  let s = c land (Array.length tr.slots - 1) in
  let p =
    match tr.slots.(s) with
    | Some p -> p
    | None ->
        let p = make_piece () in
        tr.slots.(s) <- Some p;
        p
  in
  fill tr.ctx p c;
  p

(* Twins. A chord runs both ways in a piece where the region of [hi] between two
   of the band's polygons has no area, such as a joined saddle whose in corners
   equal [hi]; the twin of a chord is then the other way in the same piece. *)

let rec chord_from p e f =
  if f = p.ne then -1
  else if
    p.e_side.(f) < 0 && p.e_src.(f) = p.e_dst.(e) && p.e_dst.(f) = p.e_src.(e)
  then f
  else chord_from p e (f + 1)

let chord_twin p e = chord_from p e 0

(* The first edge of [q] from [f] on along a side from the point of identity
   [gs] to that of [gd], or [-1]. *)
let rec side_from q gs gd f =
  if f = q.ne then -1
  else if
    q.e_side.(f) >= 0
    && q.b_gid.(q.e_src.(f)) = gs
    && q.b_gid.(q.e_dst.(f)) = gd
  then f
  else side_from q gs gd (f + 1)

(* The edge of the piece across edge [e]'s side that runs over it the other way,
   or [-1]. *)
let side_twin tr p e =
  let k = p.e_side.(e) in
  if p.nbr.(k) < 0 then -1
  else
    side_from (piece tr p.nbr.(k)) p.b_gid.(p.e_dst.(e)) p.b_gid.(p.e_src.(e)) 0

let on_domain_boundary p e =
  let k = p.e_side.(e) in
  k >= 0 && p.nbr.(k) < 0

let[@inline] kind kinds c = Char.code (Bytes.unsafe_get kinds c)

(* A cell whose piece has no edge of the band's boundary: its samples all below
   [lo], all at or above [hi], or all missing, or all in the band as are those
   of the four cells around it. [kinds] holds the class of each cell's samples
   if they all have one, and 4 otherwise. *)
let idle ctx kinds c =
  let ncx = ctx.f.nx - 1 and ncy = ctx.f.ny - 1 in
  let k = kind kinds c in
  k < 4
  && (k <> 1
     ||
     let i = c / ncx and j = c mod ncx in
     i > 0
     && i < ncy - 1
     && j > 0
     && j < ncx - 1
     && kind kinds (c - ncx) = 1
     && kind kinds (c + ncx) = 1
     && kind kinds (c - 1) = 1
     && kind kinds (c + 1) = 1)

let kinds ctx =
  let nx = ctx.f.nx in
  let ncx = nx - 1 and ncy = ctx.f.ny - 1 in
  let kinds = Bytes.create (ncx * ncy) in
  for i = 0 to ncy - 1 do
    for j = 0 to ncx - 1 do
      let tl = (i * nx) + j in
      let k = cls ctx tl in
      let same =
        k = cls ctx (tl + 1)
        && k = cls ctx (tl + nx)
        && k = cls ctx (tl + nx + 1)
      in
      Bytes.unsafe_set kinds
        ((i * ncx) + j)
        (Char.unsafe_chr (if same then k else 4))
    done
  done;
  kinds

let push tr x y on_boundary =
  if tr.n = Array.length tr.rx then begin
    let grow a = Array.append a (Array.make (Array.length a) 0.) in
    tr.rx <- grow tr.rx;
    tr.ry <- grow tr.ry;
    tr.rd <- Bytes.extend tr.rd 0 (Bytes.length tr.rd)
  end;
  tr.rx.(tr.n) <- x;
  tr.ry.(tr.n) <- y;
  Bytes.set tr.rd tr.n (if on_boundary then '\001' else '\000');
  tr.n <- tr.n + 1

let traced ctx c e = ctx.visited.(c) land (1 lsl e) <> 0

(* The ring through boundary edge [e0] of cell [c0], into [tr.rx] and
   [tr.ry]. *)
let walk tr c0 e0 =
  let ctx = tr.ctx in
  let limit = max_edges * Array.length ctx.visited in
  tr.n <- 0;
  let c = ref c0 and e = ref e0 and fin = ref false in
  while not !fin do
    let p = piece tr !c in
    ctx.visited.(!c) <- ctx.visited.(!c) lor (1 lsl !e);
    push tr p.b_x.(p.e_src.(!e)) p.b_y.(p.e_src.(!e)) (on_domain_boundary p !e);
    if tr.n > limit then failwith "Field2: a ring does not close";
    let q = ref p and next = ref p.e_next.(!e) and turning = ref true in
    while !turning do
      let qc = !q in
      let chord = qc.e_side.(!next) < 0 in
      let t = if chord then chord_twin qc !next else side_twin tr qc !next in
      if t < 0 then turning := false
      else if chord then next := qc.e_next.(t)
      else begin
        let q' = piece tr qc.nbr.(qc.e_side.(!next)) in
        q := q';
        next := q'.e_next.(t)
      end
    done;
    let qc = !q in
    if qc.cell = c0 && !next = e0 then fin := true
    else if traced ctx qc.cell !next then
      failwith "Field2: a ring reaches an edge already traced"
    else begin
      c := qc.cell;
      e := !next
    end
  done

let trace f ~lo ~hi emit =
  let n = f.nx * f.ny in
  let cls = Bytes.create n in
  for k = 0 to n - 1 do
    let v = Array.unsafe_get f.z k in
    Bytes.unsafe_set cls k
      (if not (Float.is_finite v) then '\003'
       else if v < lo then '\000'
       else if v < hi then '\001'
       else '\002')
  done;
  let ncells = (f.nx - 1) * (f.ny - 1) in
  let rec slots k =
    if k >= Int.min ((2 * f.nx) + 2) ncells then k else slots (2 * k)
  in
  let ctx = { f; lo; hi; cls; visited = Array.make ncells 0 } in
  let tr =
    {
      ctx;
      slots = Array.make (slots 1) None;
      n = 0;
      rx = Array.make 64 0.;
      ry = Array.make 64 0.;
      rd = Bytes.create 64;
    }
  in
  let kinds = kinds ctx in
  for c = 0 to ncells - 1 do
    if not (idle ctx kinds c) then
      for e = 0 to (piece tr c).ne - 1 do
        let p = piece tr c in
        if
          (not (traced ctx c e))
          &&
          if p.e_side.(e) < 0 then chord_twin p e < 0 else side_twin tr p e < 0
        then begin
          walk tr c e;
          emit tr
        end
      done
  done

(* The [len] points of the ring in [tr] from point [first], in the plane's
   positive order. *)
let points f tr first len =
  let n = tr.n and mirrored = mirrored f in
  let at q = (first + if mirrored then len - 1 - q else q) mod n in
  ( Array.init len (fun q -> tr.rx.(at q)),
    Array.init len (fun q -> tr.ry.(at q)) )

(* Contours *)

let isoband ~lo ~hi f =
  if Float.is_nan lo || Float.is_nan hi then
    invalid_arg "Field2.isoband: a bound is NaN";
  if lo > hi then
    invalid_arg (Printf.sprintf "Field2.isoband: lo %g is above hi %g" lo hi);
  if lo = hi || f.nx < 2 || f.ny < 2 then Pgon2.v []
  else begin
    let rings = ref [] in
    trace f ~lo ~hi (fun tr ->
        let xs, ys = points f tr 0 tr.n in
        rings := Ring2.v xs ys :: !rings);
    Pgon2.v (List.rev !rings)
  end

(* The curves of the ring in [tr]: the runs of its edges off the domain's
   boundary, from the first edge after one on it. *)
let curves f tr path =
  let n = tr.n in
  let on_boundary i = Bytes.get tr.rd (i mod n) = '\001' in
  let rec first_on i =
    if i = n then -1 else if on_boundary i then i else first_on (i + 1)
  in
  match first_on 0 with
  | -1 ->
      let xs, ys = points f tr 0 n in
      Path.append (Path.polygon xs ys) path
  | b ->
      let path = ref path and run = ref 0 in
      for k = 1 to n do
        if not (on_boundary (b + k)) then incr run
        else if !run > 0 then begin
          let xs, ys = points f tr (b + k - !run) (!run + 1) in
          path := Path.append (Path.polyline xs ys) !path;
          run := 0
        end
      done;
      !path

let isoline l f =
  if Float.is_nan l then invalid_arg "Field2.isoline: the level is NaN";
  if f.nx < 2 || f.ny < 2 then Path.empty
  else begin
    let path = ref Path.empty in
    trace f ~lo:l ~hi:infinity (fun tr -> path := curves f tr !path);
    !path
  end
