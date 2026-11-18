(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let fold f l = List.fold_left f (List.hd l) (List.tl l)

(* Points *)

(* The centres pass through the longitude and latitude of [dst]'s own datum,
   so the first transform takes no datum step, and every point [crs]'s
   projection refuses is no data. *)
let onward crs pivot ll =
  let inside = Crs.in_domain ~src:pivot ~dst:crs ll in
  let shape = Nx.shape ll in
  let inside = Nx.reshape [| shape.(0); shape.(1); 1 |] inside in
  let ll = Nx.where inside ll (Nx.full_like ll Float.nan) in
  Crs.transform ~src:pivot ~dst:crs ll

let into crs (dst : Grid.t) c =
  if Crs.equal crs dst.crs then c
  else
    let pivot = { dst.crs with Crs.projection = None } in
    onward crs pivot (Crs.transform ~src:dst.crs ~dst:pivot c)

let centres ~at crs dst = into crs dst (Grid.centres ~at dst)

(* A multi-index of an array, as a message prints it. *)
let index i = String.concat ", " (List.map string_of_int (Array.to_list i))

(* [dst]'s pixel points (i, j), for i ≤ rows and j ≤ cols, as [Grid.world]
   computes them. The area rules move a footprint's corners to within 180°
   of longitude of its first, so a cell with a corner 180° or more from its
   first, by [dst]'s own longitudes, is refused. On a geographic [dst] they
   are [Grid.world]'s, unwrapped. On a projection they are λ - λ₀ in [-180,
   180], on the side of the cut that the easting's sign gives, as the
   inverse of the transverse Mercator, Pseudo-Mercator and the sinusoid
   gives it. Each of those takes its domain to within 180° of λ₀, so a cell
   spans 180° only across the domain's edge or near a pole. Albers' cut,
   λ₀ ± 180°, is a ray from its apex inside its plane, which a valid cell
   can cross, so Albers is not checked; its cells that hold a pole wind
   around it, which [place] refuses. Nor are equal projected systems, whose
   corners are their cells' own and move by no turn. A projection whose
   plane holds a pole as an interior point, as a polar stereographic or
   azimuthal one does, makes cells that hold it without a corner far from
   the others, so it needs a test of its own here, of whether a cell holds
   the pole's image.

   The transform of a projected [dst]'s corners to its datum's longitude
   and latitude refuses a corner outside [dst.crs]'s domain in
   [Crs.transform]'s name, at the corner's index. On the host that refusal
   raises at the call, and it is raised again in [corners]' name, with no
   second transform. On another placement it raises at a wait
   ([Nx.check]), after [corners] has returned, and keeps
   [Crs.transform]'s name. *)
let corners ~at crs (dst : Grid.t) =
  let along n = Nx.arange_f_at at Nx.float64 0. (float_of_int (n + 1)) 1. in
  let shape = [| dst.rows + 1; dst.cols + 1 |] in
  let axis n s = Nx.broadcast_to shape (Nx.reshape s (along n)) in
  let r = axis dst.rows [| dst.rows + 1; 1 |]
  and c = axis dst.cols [| 1; dst.cols + 1 |] in
  let c = Grid.world dst (Nx.stack ~axis:2 [ r; c ]) in
  let x p = Nx.slice [ A; A; I 0 ] p in
  let half_turns lon =
    let corner (i, j) =
      Nx.slice [ R (i, i + dst.rows); R (j, j + dst.cols) ] lon
    in
    let far ij =
      Nx.greater_equal_s (Nx.abs (Nx.sub (corner ij) (corner (0, 0)))) 180.
    in
    let far = fold Nx.logical_or (List.map far [ (0, 1); (1, 1); (1, 0) ]) in
    Nx.check Nx.Ptree.unit (Nx.logical_not far) () (fun i () ->
        Invalid_argument
          (strf
             "Jord.Warp.corners: cell [%s] has a corner 180° or more of \
              longitude from its first; split it into cells narrower than \
              180°"
             (index i)))
  in
  match dst.crs.projection with
  | None ->
      half_turns (x c);
      into crs dst c
  | Some _ when Crs.equal crs dst.crs -> c
  | Some projection ->
      let pivot = { dst.crs with Crs.projection = None } in
      let ll =
        let refused = "Jord.Crs.transform: point " in
        let n = String.length refused in
        try Crs.transform ~src:dst.crs ~dst:pivot c
        with Invalid_argument m when String.starts_with ~prefix:refused m ->
          (* The refusal up to transform's remedy, which this one replaces. *)
          let e =
            Option.value ~default:(String.length m) (String.index_opt m ';')
          in
          invalid_arg
            (strf
               "Jord.Warp.corners: corner %s; dst's corners must lie in its \
                system's domain (Crs.in_domain), and around a pole dst is a \
                geographic cap of cells narrower than 180°"
               (String.sub m n (e - n)))
      in
      (match projection with
      | Crs.Albers _ -> ()
      | Crs.Transverse_mercator { lon0; false_easting; _ }
      | Crs.Pseudo_mercator { lon0; false_easting; _ }
      | Crs.Sinusoidal { lon0; false_easting; _ } ->
          let d = Nx.sub_s (x ll) lon0 in
          let d = Nx.sub d (Nx.mul_s (Nx.round (Nx.div_s d 360.)) 360.) in
          half_turns
            (Nx.mul (Nx.abs d) (Nx.sign (Nx.sub_s (x c) false_easting))));
      onward crs pivot ll

(* A product that OCaml's arm64 backend cannot fuse into the addition that uses
   it, as [Grid]'s. *)
let mul a b = Sys.opaque_identity (a *. b)

(* The least, or by [Float.max] the greatest, longitude of [g]'s four
   corners, as [Grid.world] computes it. *)
let extreme f (g : Grid.t) =
  let ox, _ = g.origin and rx, _ = g.row_step and cx, _ = g.col_step in
  let r = float_of_int g.rows and c = float_of_int g.cols in
  let lon r c = mul r rx +. mul c cx +. ox in
  f (f (lon 0. 0.) (lon r 0.)) (f (lon 0. c) (lon r c))

let west = extreme Float.min

let pixel (g : Grid.t) x =
  match g.crs.projection with
  | None ->
      let batch, lon, lat = Points.split "Jord.Warp.pixel" x in
      let w = west g in
      let turns = Nx.floor (Nx.div_s (Nx.sub_s lon w) 360.) in
      let lon = Nx.sub lon (Nx.mul_s turns 360.) in
      Grid.pixel g (Points.join batch lon lat)
  | Some _ -> Grid.pixel g x

(* One turn of longitude as a displacement of [g]'s pixel points, by the inverse
   of [g]'s map. *)
let turn (g : Grid.t) =
  let rx, ry = g.row_step and cx, cy = g.col_step in
  let d = mul rx cy -. mul cx ry in
  (360. *. cy /. d, -360. *. ry /. d)

(* Whether [g] is a geographic grid whose columns span one turn, a turn of
   [(0, ±cols)] pixels to rounding. It holds every longitude once, so the
   rules read its columns modulo [cols], as [pixel] wraps longitudes, and a
   footprint or a bilinear point across its west edge reads both sides. The
   double nearest 360° / cols as the step gives a turn within 4 u cols of
   cols, to first order in u = 2⁻⁵³, from the step's rounding and three of
   [turn]'s, so the bound is 8 u cols. A turn's row is 0 exactly when the
   column step's latitude is 0. *)
let wraps (g : Grid.t) =
  let tr, tc = turn g and n = float_of_int g.cols in
  g.crs.projection = None
  && tr = 0.
  && Float.abs (Float.abs tc -. n) <= n *. 0x1p-50

(* A column of [n], of float index, modulo [n], in [0, n). *)
let modulo n v =
  let n = float_of_int n in
  Nx.sub v (Nx.mul_s (Nx.floor (Nx.div_s v n)) n)

(* Samplers *)

(* [x]'s rows and columns and its other axes, after checking [valid]: the
   array's first two axes, then any axes of size 1, one value per cell
   either way, so Band.decode's mask of a one-channel image serves. *)
let axes fn ?valid x =
  let shape = Nx.shape x in
  let rank = Array.length shape in
  if rank < 2 then
    invalid_arg (strf "%s: the array's rank %d is below 2" fn rank);
  let rows = shape.(0) and cols = shape.(1) in
  let per_cell s =
    let n = Array.length s in
    n >= 2 && s.(0) = rows && s.(1) = cols
    && Array.for_all (fun d -> d = 1) (Array.sub s 2 (n - 2))
  in
  (match valid with
  | Some v when not (per_cell (Nx.shape v)) ->
      invalid_arg
        (strf
           "%s: the mask's shape is not the array's first two axes, then \
            axes of size 1"
           fn)
  | _ -> ());
  (rows, cols, Array.sub shape 2 (rank - 2))

(* [x]'s rows, columns and other axes, as [axes], after checking that its
   first two are a block of [src]'s cells from [(row, col)], its columns
   modulo [src.cols] where [wraps]. *)
let block fn ~wraps ~row ~col ?valid (src : Grid.t) x =
  let brows, bcols, rest = axes fn ?valid x in
  let rows = src.rows and cols = src.cols in
  if
    row < 0 || col < 0
    || row + brows > rows
    || (if wraps then bcols > cols || (bcols > 0 && col >= cols)
        else col + bcols > cols)
  then
    invalid_arg
      (strf "%s: the block of %d × %d cells from (%d, %d) is not in the \
             grid's %d × %d"
         fn brows bcols row col rows cols);
  (brows, bcols, rest)

(* Whether cell (i, j), of float indices, lies in the [rows × cols] grid.
   Each axis is checked alone, since a flat index of column -1 or [cols]
   lands in the grid. NaN is in no cell. *)
let within ~rows ~cols i j =
  let along v n = Nx.logical_and (Nx.greater_equal_s v 0.) (Nx.less_s v n) in
  Nx.logical_and (along i (float_of_int rows)) (along j (float_of_int cols))

(* Whether cell (i, j) lies in the grid, as [within], and its flat index
   there, -1 elsewhere, so that NaN is never cast. *)
let cell ~rows ~cols i j =
  let inside = within ~rows ~cols i j in
  let k = Nx.add (Nx.mul_s i (float_of_int cols)) j in
  (inside, Nx.cast Nx.int64 (Nx.where inside k (Nx.full_like k (-1.))))

(* Cell (i, j) of a grid in the block of [rows × cols] cells from [(row,
   col)], as [cell] there, its column from [col] taken by [wrap]. *)
let local ?(wrap = Fun.id) ~row ~col ~rows ~cols i j =
  cell ~rows ~cols
    (Nx.sub_s i (float_of_int row))
    (wrap (Nx.sub_s j (float_of_int col)))

(* [x] as one row per cell, [[rows · cols; ...]], and whether the cells of
   flat indices [k], [inside] the grid, have data. *)
let cells ~rows ~cols rest ?valid x =
  let data =
    match valid with
    | None -> fun _ inside -> inside
    | Some v ->
        let v = Nx.reshape [| rows * cols |] v in
        fun k _ -> Nx.take k v
  in
  (Nx.reshape (Array.append [| rows * cols |] rest) x, data)

(* [m] broadcast over [rank] more trailing axes. *)
let lift rank m = Nx.reshape (Array.append (Nx.shape m) (Array.make rank 1)) m

(* A mask of [batch] in its values' shape, [batch] then [rest], the same
   along [rest]. *)
let widen batch rest m =
  Nx.broadcast_to (Array.append batch rest)
    (lift (Array.length rest) (Nx.reshape batch m))

(* Raises through [Nx.check] at the first point, of the batch shape
   [batch], that reads a cell of [src] outside the block of [rows × cols]
   cells, its message ending in [remedy]. [taps] are the cells (i, j) it
   reads, each with whether it lies in the block, and the point must also
   have its [containing] cell in [src] where that is not its one tap. The
   block of the whole [src] holds every cell. *)
let outside fn ~remedy ?containing (src : Grid.t) ~rows ~cols batch taps =
  if rows < src.rows || cols < src.cols then
    let in_src (i, j) = within ~rows:src.rows ~cols:src.cols i j in
    let out (i, j, inside) =
      Nx.logical_and (in_src (i, j)) (Nx.logical_not inside)
    in
    let bad = fold Nx.logical_or (List.map out taps) in
    let bad =
      match containing with
      | None -> bad
      | Some c -> Nx.logical_and (in_src c) bad
    in
    Nx.check Nx.Ptree.unit
      (Nx.logical_not (Nx.reshape batch bad))
      ()
      (fun i () ->
        Invalid_argument
          (strf "%s: point [%s] reads a cell outside the block; %s" fn
             (index i) remedy))

let nearest ?(row = 0) ?(col = 0) ?valid src p x =
  let fn = "Jord.Warp.nearest" in
  let rows, cols, rest = block fn ~wraps:false ~row ~col ?valid src x in
  let batch, r, c = Points.split fn p in
  let x, data = cells ~rows ~cols rest ?valid x in
  let i = Nx.floor r and j = Nx.floor c in
  let inside, k = local ~row ~col ~rows ~cols i j in
  let remedy = "read the block Warp.window Nearest gives" in
  outside fn ~remedy src ~rows ~cols batch [ (i, j, inside) ];
  let ok = data k inside in
  let v = Nx.take ~axis:0 k x in
  let v = Nx.where (lift (Array.length rest) ok) v (Nx.zeros_like v) in
  (Nx.reshape (Array.append batch rest) v, widen batch rest ok)

(* How a float64 mean is stored in [dtype]: as it is for floats, which the
   cast rounds once; as ⌊v + 1/2⌋ for integers, GDAL's rule on every sign,
   whose integral result the cast keeps. *)
let rounding fn (type v s) (dtype : (v, s) Nx.dtype) =
  match Nx.Dtype.kind dtype with
  | Float -> Fun.id
  | Signed | Unsigned -> fun v -> Nx.floor (Nx.add_s v 0.5)
  | Complex | Boolean ->
      let name = Nx.Dtype.name dtype in
      invalid_arg (strf "%s: %s is not a real dtype" fn name)

let bilinear ?(row = 0) ?(col = 0) ?valid src p x =
  let fn = "Jord.Warp.bilinear" in
  let round = rounding fn (Nx.dtype x) in
  (* On a source that spans 360°, the taps' columns are read modulo its
     columns, as the area rules read them, and so are the block's. *)
  let wraps = wraps src in
  let wrap = if wraps then modulo src.cols else Fun.id in
  let rows, cols, rest = block fn ~wraps ~row ~col ?valid src x in
  let batch, r, c = Points.split fn p in
  let rank = Array.length rest in
  let dtype = Nx.dtype x in
  let x, data = cells ~rows ~cols rest ?valid x in
  (* The four cells around (r, c) are those whose centres bound it: rows i0
     and i0 + 1 from u = r - 1/2, weighted 1 - a and a, and likewise for
     columns. *)
  let u = Nx.sub_s r 0.5 and v = Nx.sub_s c 0.5 in
  let i0 = Nx.floor u and j0 = Nx.floor v in
  let a = Nx.sub u i0 and b = Nx.sub v j0 in
  let along i0 a = [ (i0, Nx.rsub_s 1. a); (Nx.add_s i0 1., a) ] in
  (* Each tap, its column taken by [wrap], its weight, and whether it lies
     in the block and its index there. *)
  let around =
    List.concat_map
      (fun (i, wi) ->
        List.map
          (fun (j, wj) ->
            (i, wrap j, Nx.mul wi wj, local ~wrap ~row ~col ~rows ~cols i j))
          (along j0 b))
      (along i0 a)
  in
  let i = Nx.floor r and j = Nx.floor c in
  let remedy =
    if wraps && col + cols = src.cols && cols < src.cols then
      "read the block Warp.window Bilinear gives, in two parts where its \
       columns pass the last"
    else "read the block Warp.window Bilinear gives"
  in
  outside fn ~remedy ~containing:(i, j) src ~rows ~cols batch
    (List.map (fun (i, j, _, (inside, _)) -> (i, j, inside)) around);
  (* A tap left out is selected away, never multiplied by 0: NaN · 0 is NaN,
     in the value and in the gradient, so its value is replaced by 0 before
     the product and the product selected away after it. A tap of weight 0
     whose value is finite is kept, adding 0 to the value but its weight's
     derivative to the gradient: on a centre line, where a neighbour's weight
     is 0, the mean then moves toward that neighbour. Whether a value is
     finite differs from channel to channel, so the weights are per
     channel. *)
  let tap (_, _, w, (inside, k)) =
    let xk = Nx.cast Nx.float64 (Nx.take ~axis:0 k x) in
    let w = Nx.broadcast_to (Nx.shape xk) (lift rank w) in
    let use =
      Nx.logical_and
        (lift rank (data k inside))
        (Nx.logical_or (Nx.greater_s w 0.) (Nx.isfinite xk))
    in
    let wx = Nx.mul w (Nx.where use xk (Nx.zeros_like xk)) in
    (Nx.where use wx (Nx.zeros_like wx), Nx.where use w (Nx.zeros_like w))
  in
  let taps = List.map tap around in
  let total l = List.fold_left Nx.add (List.hd l) (List.tl l) in
  let sum = total (List.map fst taps) and weight = total (List.map snd taps) in
  let inside, k = local ~wrap ~row ~col ~rows ~cols i j in
  let inside =
    if not wraps then inside
    else Nx.logical_and inside (within ~rows:src.rows ~cols:src.cols i j)
  in
  let ok = data k inside in
  let mean = Nx.div sum weight in
  let mean = Nx.where (lift rank ok) mean (Nx.zeros_like mean) in
  ( Nx.reshape (Array.append batch rest) (Nx.cast dtype (round mean)),
    widen batch rest ok )

(* Area rules *)

(* Twice the signed area of the polygon of [corners], [(row, col)], positive
   when it turns from the column axis to the row axis, by the shoelace formula
   about its first corner. *)
let shoelace corners =
  let y0, x0 = List.hd corners in
  let rel = List.map (fun (y, x) -> (Nx.sub y y0, Nx.sub x x0)) corners in
  let next = List.tl rel @ [ List.hd rel ] in
  let cross (y, x) (y', x') = Nx.sub (Nx.mul x y') (Nx.mul x' y) in
  let terms = List.map2 cross rel next in
  List.fold_left Nx.add (List.hd terms) (List.tl terms)

(* The area of the polygon of [corners] inside the cell [0, 1] × [0, 1], its
   [(row, col)] relative to the cell's corner, and the bound below which that
   area is rounding: the sum of {!Overlap.term} over its edges, on the cell's
   whole column. The sum has the sign opposite to [orient], the polygon's
   orientation, which turns it to the area; rounding leaves an area below the
   bound of either sign.

   In a cell the polygon does not meet, the exact sum is 0 and the computed
   one is at most 24 u M W, with u = 2⁻⁵³, W the sum of the widths' magnitudes
   and M the largest corner coordinate's magnitude, at least 1, to first
   order: each term's (15M + 5) u |w| and the sum's 3 u W, (15M + 8) u W in
   all, at most 23 u M W. *)
let overlap orient corners =
  let y0, _ = List.hd corners in
  let lo = Nx.scalar_like y0 0. and hi = Nx.scalar_like y0 1. in
  let next = List.tl corners @ [ List.hd corners ] in
  let edges = List.map2 (Overlap.term ~lo ~hi) corners next in
  let total l = List.fold_left Nx.add (List.hd l) (List.tl l) in
  let s = Nx.mul (Nx.neg orient) (total (List.map fst edges)) in
  let size (y, x) = Nx.maximum (Nx.abs y) (Nx.abs x) in
  let sizes = List.map size corners in
  let m = List.fold_left Nx.maximum (Nx.ones_like s) sizes in
  let u = epsilon_float /. 2. in
  let bound = Nx.mul (Nx.mul_s m (24. *. u)) (total (List.map snd edges)) in
  (s, bound)

(* The corners [q], pixel points of [src], as footprints: their [r × c]
   shape and number, each one's four corners in order around it as [[n; 1]]
   rows and columns, a footprint with a corner that is not finite made one
   point at (0, 0), each placed by whole turns on a geographic [src]. Its
   span is the cells from the one holding its least row and column, [(r0,
   c0)], to the one holding its greatest, [(r1, c1)], clipped to [src] but
   for the columns of a source read modulo their number ([wraps]). [meets]
   holds for a footprint with finite corners whose span, before the clip,
   has a cell in [src]: the footprints that read cells. *)
type 'd placed = {
  shape : int array;
  n : int;
  rs : 'd Nx.float64_t list;
  cs : 'd Nx.float64_t list;
  r0 : 'd Nx.float64_t;
  r1 : 'd Nx.float64_t;
  c0 : 'd Nx.float64_t;
  c1 : 'd Nx.float64_t;
  wraps : bool;
  meets : 'd Nx.bool_t;
}

(* The remedy for a footprint across the west edge of a geographic [src]
   that holds its other side and does not span 360°. Where its turn is a
   whole number [n] of columns along a row, to [wraps]'s bound, [src]
   spans 360° once cut or padded to [n] columns. Otherwise no whole number
   of its columns makes a turn, and the grid cannot tell a step printed
   rounded, as 1° printed 0.9999999°, from a step of its own, as 0.7°:
   stated as 360° / n the first spans 360°, and the second would move its
   cells, by 0.2° at 514 columns of 0.7°. So the message gives both: the
   step restated where it is a rounding, and otherwise a cut narrower than
   a turn that holds each footprint whole, or a [dst] whose cells keep off
   the edge. *)
let seam_remedy (src : Grid.t) =
  let tr, tc = turn src in
  let n = Float.round (Float.abs tc) in
  let k = int_of_float n in
  if tr = 0. && Float.abs (Float.abs tc -. n) <= n *. 0x1p-50 then
    if k < src.cols then
      strf "Grid.window it to its first turn, %d columns, and slice x to them"
        k
    else strf "pad it to one turn, %d columns, with cells not valid" k
  else
    "no whole number of its columns makes a turn: where its column step \
     rounds (±360° / n, 0) for a whole n, state that step; otherwise \
     Grid.window it, and slice x, to columns narrower than a turn that hold \
     every footprint whole, or make dst with no cell across its west edge"

let place fn (src : Grid.t) q =
  let r, c =
    match Nx.shape q with
    | [| r; c; 2 |] when r > 0 && c > 0 -> (r - 1, c - 1)
    | _ ->
        invalid_arg (strf "%s: the corners' shape is not [r + 1; c + 1; 2]" fn)
  in
  let n = r * c in
  let corner (i, j) k =
    Nx.reshape [| n; 1 |] (Nx.slice [ R (i, i + r); R (j, j + c); I k ] q)
  in
  let around = [ (0, 0); (0, 1); (1, 1); (1, 0) ] in
  let rs = List.map (fun ij -> corner ij 0) around
  and cs = List.map (fun ij -> corner ij 1) around in
  (* A footprint with a corner that is not finite, NaN outside the source's
     domain, has no data. Its corners become one point, of area 0 everywhere, so
     that the NaN reaches neither the result nor the gradient. *)
  let finite =
    List.fold_left
      (fun acc v -> Nx.logical_and acc (Nx.isfinite v))
      (Nx.isfinite (List.hd rs))
      (List.tl rs @ cs)
  in
  let clean v = Nx.where finite v (Nx.zeros_like v) in
  let rs = List.map clean rs and cs = List.map clean cs in
  (* [pixel] takes each corner of a geographic source into the 360° from its
     west corner, so a footprint across that meridian has corners a turn apart.
     Each corner moves by whole turns to within 180° of longitude of the
     footprint's first, then the footprint by whole turns so that its easternmost corner
     lies in the 360° east of the west corner, closed on the east: across that
     meridian, it reads the side east of it. *)
  let geographic = src.crs.projection = None in
  let tr, tc = turn src in
  let (ox, _), (rx, _), (cx, _) = (src.origin, src.row_step, src.col_step) in
  (* A pixel point's longitude east of [src]'s origin, in turns. *)
  let along r c = Nx.div_s (Nx.add (Nx.mul_s r rx) (Nx.mul_s c cx)) 360. in
  let west = (west src -. ox) /. 360. in
  let rs, cs =
    if not geographic then (rs, cs)
    else
      let by t r c = (Nx.sub r (Nx.mul_s t tr), Nx.sub c (Nx.mul_s t tc)) in
      let a0 = along (List.hd rs) (List.hd cs) in
      let rs, cs =
        List.split
          (List.map2
             (fun r c -> by (Nx.round (Nx.sub (along r c) a0)) r c)
             rs cs)
      in
      let east = fold Nx.maximum (List.map2 along rs cs) in
      let t = Nx.sub_s (Nx.ceil (Nx.sub_s east west)) 1. in
      List.split (List.map2 (by t) rs cs)
  in
  let wraps = wraps src in
  (* The cells outside [src] have no data, so the span is clipped to it, but
     for the columns of a source read modulo their number. *)
  let lo l = Nx.floor (fold Nx.minimum l)
  and hi l = Nx.floor (fold Nx.maximum l) in
  let rlo = lo rs and rhi = hi rs and clo = lo cs and chi = hi cs in
  let has lo hi n =
    Nx.logical_and (Nx.less_s lo (float_of_int n)) (Nx.greater_equal_s hi 0.)
  in
  let rows_meet = Nx.logical_and finite (has rlo rhi src.rows) in
  let meets =
    if wraps then rows_meet
    else Nx.logical_and rows_meet (has clo chi src.cols)
  in
  if geographic then begin
    let refuse bad what =
      Nx.check Nx.Ptree.unit
        (Nx.logical_not (Nx.reshape [| r; c |] bad))
        ()
        (fun i () ->
          Invalid_argument (strf "%s: footprint [%s] %s" fn (index i) what))
    in
    (* Placed, a footprint's corners lie within 180° of longitude of its
       first. Where each of its intended edges spans less than 180°, its
       placed edges are those edges exactly when they span less than 180°
       too: a placed edge is its intended edge moved by a whole number of
       turns, below one in magnitude, so by none. A footprint around a pole
       has a placed edge of 180° or more, since its edges wind once around
       the circle, and so has one with a corner 180° or more from its
       first. Its rows do not move with turns where a turn's row is 0, so a
       footprint whose rows miss [src] reads nothing and passes. *)
    let a = List.map2 along rs cs in
    let next = List.tl a @ [ List.hd a ] in
    let long a b = Nx.greater_equal_s (Nx.abs (Nx.sub b a)) 0.5 in
    refuse
      (Nx.logical_and rows_meet (fold Nx.logical_or (List.map2 long a next)))
      "has an edge of 180° of longitude or more, or winds around a pole; \
       split a geographic cell into cells narrower than 180°, and around a \
       pole make dst a geographic cap of such cells, since no projected \
       cell around a pole reads whole";
    (* A footprint across the west edge of a source that does not wrap is
       placed east of the edge, and reads that side alone. Where its least
       longitude, a turn east, still lies west of the source's east corner,
       the source holds its part west of the edge there, a turn east. *)
    if not wraps then
      let east = (extreme Float.max src -. ox) /. 360. in
      let lo = fold Nx.minimum a in
      refuse
        (Nx.logical_and meets
           (Nx.logical_and (Nx.less_s lo west)
              (Nx.less_s (Nx.add_s lo 1.) east)))
        (strf
           "lies across the west edge of a source that holds its other side \
            a turn east, and the source does not span 360°: its turn is \
            %.10g columns and %.10g rows, against its %d columns; %s"
           tc (tr +. 0.) src.cols (seam_remedy src))
  end;
  let clip n v = Nx.clamp ~min:0. ~max:(float_of_int (max 0 (n - 1))) v in
  let c0, c1 =
    if wraps then (clo, chi) else (clip src.cols clo, clip src.cols chi)
  in
  {
    shape = [| r; c |];
    n;
    rs;
    cs;
    r0 = clip src.rows rlo;
    r1 = clip src.rows rhi;
    c0;
    c1;
    wraps;
    meets;
  }

(* The most rows or columns each footprint's span crosses, [r × c]: one less
   than the [k] it needs, and 0 for a footprint that reads no cell. *)
let crossed p =
  let e = Nx.maximum (Nx.sub p.r1 p.r0) (Nx.sub p.c1 p.c0) in
  Nx.reshape p.shape (Nx.where p.meets e (Nx.zeros_like e))

let span src q =
  let p = place "Jord.Warp.span" src q in
  if p.n = 0 then 1 else int_of_float (Nx.item [] (Nx.max (crossed p))) + 1

(* The least run of columns, [(col, cols)] with [col < n], that holds the
   arcs [(s, e)] of the circle of [n] columns, each from [s] in [0, n) to
   [e], one past its last: the circle less its widest gap. Sorted by [s],
   the cover so far starts as the arcs' wrapped tail, max e - n; each arc
   then leaves the gap from the cover to its [s], and the circle closes with
   the gap from the last cover to the first [s] a turn on. *)
let least_run n arcs =
  let nf = float_of_int n in
  let arcs = List.sort compare arcs in
  let e = List.fold_left (fun m (_, e) -> Float.max m e) 0. arcs in
  let first = fst (List.hd arcs) in
  let widest, start, _ =
    List.fold_left
      (fun (widest, start, cover) (s, e) ->
        let gap = s -. cover in
        let widest, start =
          if gap > widest then (gap, s) else (widest, start)
        in
        (widest, start, Float.max cover e))
      (first +. nf -. e, first, e -. nf)
      arcs
  in
  if widest <= 0. then (0, n)
  else (int_of_float start, n - int_of_float widest)

(* The arc of the columns [c0] to [c1] on the circle of [n] columns. *)
let arc n c0 c1 =
  let s = c0 -. (Float.floor (c0 /. float_of_int n) *. float_of_int n) in
  (s, s +. c1 -. c0 +. 1.)

(* The least block of the cells the footprints that read cells span: the
   least and greatest of their rows and, on a source not read modulo its
   columns, of their columns; on one that is, the least run of columns that
   holds each span's. *)
let area_window (g : Grid.t) q =
  let p = place "Jord.Warp.area_window" g q in
  let get v = Nx.to_array (Nx.reshape [| p.n |] v) in
  let meets = Nx.to_array (Nx.reshape [| p.n |] p.meets) in
  let r0 = get p.r0 and r1 = get p.r1 and c0 = get p.c0 and c1 = get p.c1 in
  let read = List.filter (fun i -> meets.(i)) (List.init p.n Fun.id) in
  if read = [] then (0, 0, 0, 0)
  else
    let over f a = List.fold_left (fun m i -> f m a.(i)) a.(List.hd read) in
    let least a = over Float.min a read and most a = over Float.max a read in
    let row = int_of_float (least r0) in
    let rows = int_of_float (most r1) - row + 1 in
    if not p.wraps then
      let col = int_of_float (least c0) in
      (row, col, rows, int_of_float (most c1) - col + 1)
    else
      let col, cols =
        least_run g.cols (List.map (fun i -> arc g.cols c0.(i) c1.(i)) read)
      in
      (row, col, rows, cols)

(* Each footprint of the corners [q] and its overlap with the cells of [x]
   that may meet it: the [k × k] cells from the one holding its least row and
   column, which hold its span, a footprint whose edge lies on a cell's edge
   reaching that one with area 0. [x] is the block of [src]'s cells from
   [(row, col)], its columns modulo [src.cols] where they wrap, and the
   footprints' geometry is [src]'s whole: only the index of a cell in [x]
   depends on the block. A footprint whose span is past [k], or that reads a
   cell outside the block, raises through [Nx.check]. The work is [n k²]
   candidate cells for [n] footprints, a few hundred float64 operations each,
   with several float64 arrays of [n k²] alive together. *)
type ('v, 's, 'd) footprints = {
  shape : int array;  (** The footprints' rows and columns. *)
  n : int;  (** The number of footprints. *)
  rest : int array;  (** [x]'s other axes. *)
  area : 'd Nx.float64_t;  (** Of each footprint in each cell, [[n; k²]]. *)
  reached : 'd Nx.bool_t;  (** Whether that area passes rounding's bound. *)
  data : 'd Nx.bool_t;  (** Whether that cell has data, [[n; k²]]. *)
  values : ('v, 's, 'd) Nx.t;  (** Its values, [[n; k²; rest]]. *)
  fraction : 'd Nx.float64_t;  (** Of each footprint with data, [[n]]. *)
  ok : 'd Nx.bool_t;  (** Whether that fraction is positive. *)
}

let footprints fn ~k ?(row = 0) ?(col = 0) ?valid (src : Grid.t) q x =
  if k < 1 then
    invalid_arg (strf "%s: k = %d is below 1; use Warp.span for k" fn k);
  let p = place fn src q in
  let brows, bcols, rest = block fn ~wraps:p.wraps ~row ~col ?valid src x in
  let cols = src.cols in
  let n = p.n and rs = p.rs and cs = p.cs in
  let e = crossed p in
  Nx.check Nx.Ptree.tensor
    (Nx.less_s e (float_of_int k))
    e
    (fun i e ->
      Invalid_argument
        (strf
           "%s: footprint [%s] spans %d cells, more than k = %d; use \
            Warp.span for k"
           fn (index i)
           (int_of_float (Nx.item [] e) + 1)
           k));
  let wrap = modulo cols in
  let held =
    let at v i = Nx.sub_s v (float_of_int i) in
    let within lo hi first n =
      Nx.logical_and (Nx.greater_equal_s (at lo first) 0.)
        (Nx.less_s (at hi first) (float_of_int n))
    in
    let rows_held = within p.r0 p.r1 row brows in
    if not p.wraps then
      Nx.logical_and rows_held (within p.c0 p.c1 col bcols)
    else if bcols >= cols then rows_held
    else
      let last = Nx.add (wrap (at p.c0 col)) (Nx.sub p.c1 p.c0) in
      Nx.logical_and rows_held (Nx.less_s last (float_of_int bcols))
  in
  let short =
    Nx.reshape p.shape (Nx.logical_and p.meets (Nx.logical_not held))
  in
  (* A block of a source read modulo its columns that stops at its last
     column, where a footprint's span runs past it, is the first of the two
     parts of a block across the west edge. *)
  let across =
    if p.wraps && bcols < cols && col + bcols = cols then
      Nx.greater_equal_s
        (Nx.add (wrap p.c0) (Nx.sub p.c1 p.c0))
        (float_of_int cols)
    else Nx.zeros_like p.meets
  in
  Nx.check
    Nx.Ptree.(pair tensor tensor)
    (Nx.logical_not short)
    (short, Nx.reshape p.shape across)
    (fun i (_, across) ->
      Invalid_argument
        (if Nx.item [] across then
           strf
             "%s: footprint [%s] reads cells outside the block, across the \
              source's west edge past the block's last column; read the \
              block Warp.area_window gives in two parts, its columns from \
              col to the last, then from 0"
             fn (index i)
         else
           strf
             "%s: footprint [%s] reads cells outside the block; read the \
              block Warp.area_window gives"
             fn (index i)));
  let m = k * k in
  let steps =
    Nx.arange_f_at (Nx.placement q) Nx.float64 0. (float_of_int k) 1.
  in
  let offsets shape =
    Nx.reshape [| 1; m |] (Nx.broadcast_to [| k; k |] (Nx.reshape shape steps))
  in
  let cr = Nx.add p.r0 (offsets [| k; 1 |])
  and cc = Nx.add p.c0 (offsets [| 1; k |]) in
  let twice = shoelace (List.combine rs cs) in
  let area, bound =
    overlap (Nx.sign twice)
      (List.map2 (fun r c -> (Nx.sub r cr, Nx.sub c cc)) rs cs)
  in
  (* A candidate has data where it lies in the block, at its index there,
     which lies in [src]; the footprints that read cells hold their spans in
     the block. *)
  let x, has = cells ~rows:brows ~cols:bcols rest ?valid x in
  let bj = Nx.sub_s cc (float_of_int col) in
  let bj = if p.wraps then wrap bj else bj in
  let inside, idx =
    cell ~rows:brows ~cols:bcols (Nx.sub_s cr (float_of_int row)) bj
  in
  let data = has idx inside in
  let values = Nx.take ~axis:0 (Nx.reshape [| n * m |] idx) x in
  let values = Nx.reshape (Array.append [| n; m |] rest) values in
  (* Rounding leaves areas below [bound] in cells a footprint does not meet.
     They add nothing to a value, but a cell must pass it to give a footprint
     data or a non-finite value. *)
  let reached = Nx.greater area bound in
  let ok = Nx.any ~axes:[ 1 ] (Nx.logical_and data reached) in
  let whole = Nx.reshape [| n |] (Nx.mul_s (Nx.abs twice) 0.5) in
  let covered = Nx.sum ~axes:[ 1 ] (Nx.where data area (Nx.zeros_like area)) in
  let safe = Nx.where ok whole (Nx.ones_like whole) in
  let fraction = Nx.minimum_s (Nx.div covered safe) 1. in
  let fraction = Nx.where ok fraction (Nx.zeros_like covered) in
  { shape = p.shape; n; rest; area; reached; data; values; fraction; ok }

let average ~k ?row ?col ?valid src q x =
  let fn = "Jord.Warp.average" in
  let round = rounding fn (Nx.dtype x) in
  let dtype = Nx.dtype x in
  let f = footprints fn ~k ?row ?col ?valid src q x in
  let rank = Array.length f.rest in
  let xk = Nx.cast Nx.float64 f.values in
  let w = Nx.broadcast_to (Nx.shape xk) (lift rank f.area) in
  (* As [bilinear]'s taps: a cell left out is selected away, and a cell the
     footprint does not reach whose value is finite is kept, adding 0 to the
     value and its area's derivative to the gradient. *)
  let use =
    Nx.logical_and (lift rank f.data)
      (Nx.logical_or (lift rank f.reached) (Nx.isfinite xk))
  in
  let zero = Nx.zeros_like w in
  let wx = Nx.mul w (Nx.where use xk zero) in
  let sum = Nx.sum ~axes:[ 1 ] (Nx.where use wx zero) in
  let weight = Nx.sum ~axes:[ 1 ] (Nx.where use w zero) in
  let ok = lift rank f.ok in
  let mean = Nx.div sum (Nx.where ok weight (Nx.ones_like weight)) in
  let mean = Nx.where ok mean (Nx.zeros_like mean) in
  ( Nx.reshape (Array.append f.shape f.rest) (Nx.cast dtype (round mean)),
    widen f.shape f.rest f.fraction )

let classes fn (type v s) (dtype : (v, s) Nx.dtype) =
  match Nx.Dtype.kind dtype with
  | Signed | Unsigned -> ()
  | Float | Complex | Boolean ->
      let name = Nx.Dtype.name dtype in
      invalid_arg
        (strf "%s: %s is not an integer dtype; cast the classes to one" fn
           name)

let mode ~k ?row ?col ?valid src q x =
  let fn = "Jord.Warp.mode" in
  classes fn (Nx.dtype x);
  let f = footprints fn ~k ?row ?col ?valid src q x in
  let rank = Array.length f.rest in
  let m = Nx.dim 1 f.area in
  let shape = Nx.shape f.values in
  let w =
    Nx.where (Nx.logical_and f.data f.reached) f.area (Nx.zeros_like f.area)
  in
  let w = Nx.broadcast_to shape (lift rank w) in
  (* Each class's area is the sum of its cells', summed between the first and
     the last of its run in the sorted values: a class's run ends where the next
     value differs, and the area before the run is carried forward from its
     start by a running maximum, since running sums never fall. The greatest
     area's first run is the least class among the ties. *)
  let v, order = Nx.sort ~axis:1 f.values in
  let run = Nx.cumsum ~axis:1 (Nx.take_along_axis ~axis:1 order w) in
  let cut a = Nx.slice [ A; R (0, m - 1) ] a
  and from1 a = Nx.slice [ A; R (1, m) ] a in
  let differ = Nx.not_equal (from1 v) (cut v) in
  let first =
    Nx.full_at (Nx.placement v) Nx.bool (Array.append [| f.n; 1 |] f.rest) true
  in
  let starts = Nx.concatenate ~axis:1 [ first; differ ]
  and ends = Nx.concatenate ~axis:1 [ differ; first ] in
  let before =
    Nx.concatenate ~axis:1
      [ Nx.zeros_like (Nx.slice [ A; R (0, 1) ] run); cut run ]
  in
  let before =
    Nx.cummax ~axis:1
      (Nx.where starts before (Nx.full_like before Float.neg_infinity))
  in
  let areas = Nx.where ends (Nx.sub run before) (Nx.full_like run (-1.)) in
  let best = Nx.argmax ~axis:1 ~keepdims:true areas in
  let v = Nx.take_along_axis ~axis:1 best v in
  let v = Nx.reshape (Array.append [| f.n |] f.rest) v in
  let v = Nx.where (lift rank f.ok) v (Nx.zeros_like v) in
  (Nx.reshape (Array.append f.shape f.rest) v, widen f.shape f.rest f.fraction)

(* Windows *)

type rule = Nearest | Bilinear

(* The block a sampler reads: along each axis, the least and the greatest
   index a point whose containing cell lies in [g] reads, its row and
   column for [Nearest] and the four cells around it for [Bilinear], clipped
   to [g]; on a [g] that spans 360°, [Bilinear]'s columns are the least run
   that holds each point's two, read modulo [g.cols] as [bilinear] reads
   them. *)
let window rule (g : Grid.t) p =
  let _, r, c = Points.split "Jord.Warp.window" p in
  let inside, _ = cell ~rows:g.rows ~cols:g.cols (Nx.floor r) (Nx.floor c) in
  if not (Nx.item [] (Nx.any inside)) then (0, 0, 0, 0)
  else
    let reads x =
      match rule with
      | Nearest -> (Nx.floor x, Nx.floor x)
      | Bilinear -> (Nx.floor (Nx.sub_s x 0.5), Nx.floor (Nx.add_s x 0.5))
    in
    (* The least and the greatest index the points read along one axis, of
       [n] cells. *)
    let span x n =
      let lo, hi = reads x in
      let lo = Nx.maximum_s lo 0.
      and hi = Nx.minimum_s hi (float_of_int (n - 1)) in
      let pick v fill = Nx.where inside v (Nx.full_like v fill) in
      let first = Nx.item [] (Nx.min (pick lo Float.infinity)) in
      let last = Nx.item [] (Nx.max (pick hi Float.neg_infinity)) in
      (int_of_float first, int_of_float last - int_of_float first + 1)
    in
    let row, rows = span r g.rows in
    let col, cols =
      if rule = Nearest || not (wraps g) then span c g.cols
      else
        let lo, hi = reads c in
        let get v = Nx.to_array (Nx.reshape [| Nx.numel v |] v) in
        let inside = get inside and lo = get lo and hi = get hi in
        let arcs = ref [] in
        Array.iteri
          (fun k ok -> if ok then arcs := arc g.cols lo.(k) hi.(k) :: !arcs)
          inside;
        least_run g.cols !arcs
    in
    (row, col, rows, cols)
