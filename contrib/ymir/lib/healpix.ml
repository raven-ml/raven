(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Copyright (C) 1997-2019 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
  Benjamin D. Wandelt, Anthony J. Banday, Matthias Bartelmann, Reza Ansari
  & Kenneth M. Ganga (healpix_bare's conversions).
  SPDX-License-Identifier: ISC AND BSD-3-Clause
  ---------------------------------------------------------------------------*)

(* The conversions between directions, cells and indices follow the HEALPix
   bare bones C library, healpix_bare.c, under the notice in
   LICENSE-healpix_bare, written as nx formulas. They are derived from it and
   are neither it nor endorsed by its authors. *)

module Quantity = Ymir_units.Quantity
module Unit = Ymir_units.Unit

let strf = Printf.sprintf

type scheme = Ring | Nested
type 'f t = { frame : 'f Frame.t; nside : int; order : int; scheme : scheme }
type 'd ints = (int64, Nx.int64_elt, 'd) Nx.t

(* Górski et al. (2005), Table 1: 64-bit indices reach nside = 2^29. *)
let max_order = 29

let v frame ~nside scheme =
  let rec order k =
    if 1 lsl k = nside then Some k
    else if k = max_order then None
    else order (k + 1)
  in
  match if nside < 1 then None else order 0 with
  | Some order -> { frame; nside; order; scheme }
  | None ->
      invalid_arg
        (strf "Healpix.v: nside %d is not a power of two from 1 to 2^29" nside)

let frame h = h.frame
let nside h = h.nside
let scheme h = h.scheme
let pixels h = 12 * h.nside * h.nside

let area h =
  let n = float_of_int h.nside in
  Quantity.v Unit.steradian (Nx.scalar Nx.float64 (Float.pi /. (3. *. n *. n)))

let i64 = Int64.of_int
let land_ x c = Nx.bitwise_and x (Nx.scalar_like x c)
let floats x = Nx.cast Nx.float64 x
let floor_int x = Nx.cast Nx.int64 (Nx.floor x)

(* Cells *)

(* A pixel's {e cell} is its base pixel [face] in [0, 12) and its place [x],
   [y] in [0, nside) on it, counted from the base pixel's southern corner, [x]
   toward the north-east and [y] toward the north-west (Górski et al. §5.2).
   Faces 0 to 3 are the northern row, 4 to 7 the equatorial, 8 to 11 the
   southern, each row eastward from longitude 0. A place [(u, v)] in [[0,
   1]²] is a continuous one, [(x, y) / nside]. *)
type 'd cell = { face : 'd ints; x : 'd ints; y : 'd ints }

(* [row f] is face [f]'s row: 0 north, 1 equatorial, 2 south. [f1 f], eq.
   (11), is the ring of its southern corner in units of nside; [f2 f], eq.
   (12), the longitude of its centre in units of π/4. *)
let row face = Nx.rshift face 2
let f1 face = Nx.add_s (row face) 2L

let f2 face =
  Nx.add_s (Nx.sub (Nx.mul_s (land_ face 3L) 2L) (land_ (row face) 1L)) 1L

(* [face_of p m] is the face between the edge lines numbered [p] and [m] of
   eq. (22), in units of nside: the equatorial face where they agree, else
   the cap face they bound. *)
let face_of p m =
  Nx.where (Nx.equal p m)
    (Nx.bitwise_or p (Nx.scalar_like p 4L))
    (Nx.where (Nx.less p m) p (Nx.add_s m 8L))

(* NESTED: [face nside² + i], [i]'s even bits [x]'s and its odd bits [y]'s,
   eqs. (13) and (14). [spread v] moves the bits of [v], below 2^32, to the
   even positions; [compress] undoes it. *)
let spread v =
  let step shift mask v = land_ (Nx.bitwise_or v (Nx.lshift v shift)) mask in
  v
  |> step 16 0x0000FFFF0000FFFFL
  |> step 8 0x00FF00FF00FF00FFL |> step 4 0x0F0F0F0F0F0F0F0FL
  |> step 2 0x3333333333333333L |> step 1 0x5555555555555555L

let compress v =
  let step shift mask v = land_ (Nx.bitwise_or v (Nx.rshift v shift)) mask in
  land_ v 0x5555555555555555L
  |> step 1 0x3333333333333333L |> step 2 0x0F0F0F0F0F0F0F0FL
  |> step 4 0x00FF00FF00FF00FFL |> step 8 0x0000FFFF0000FFFFL
  |> step 16 0x00000000FFFFFFFFL

let to_nested h c =
  Nx.add
    (Nx.lshift c.face (2 * h.order))
    (Nx.add (spread c.x) (Nx.lshift (spread c.y) 1))

let of_nested h p =
  let i = land_ p (Int64.pred (Int64.shift_left 1L (2 * h.order))) in
  {
    face = Nx.rshift p (2 * h.order);
    x = compress i;
    y = compress (Nx.rshift i 1);
  }

(* RING: rings 1 to [4 nside - 1] from the north pole, each eastward from
   its first pixel. Northern cap ring [i < nside] holds [4 i] pixels from
   index [2 i (i - 1)], eqs. (2) and (3); each ring between the caps holds [4
   nside], eqs. (6) and (7); the southern cap mirrors the northern. A ring
   between the caps starts half a pixel east of longitude 0 when [i - nside]
   is even, as every cap ring does, and at longitude 0 when it is odd. *)

(* [isqrt n] is the integer square root of [n] in [0, 2^63): float64's root
   is within one of it, and one step each way corrects it. *)
let isqrt n =
  let r = floor_int (Nx.sqrt (floats n)) in
  let r = Nx.where (Nx.greater (Nx.mul r r) n) (Nx.sub_s r 1L) r in
  let r1 = Nx.add_s r 1L in
  Nx.where (Nx.less_equal (Nx.mul r1 r1) n) r1 r

(* A cell's ring [i] and its place [j] from 1 along it: in a cap, [n] = [i]
   or [4 nside - i] rings from the pole and [j = (F2 n + x - y + 1) / 2];
   between the caps the same at [n = nside], plus the ring's shift. *)
let to_ring h c =
  let ns = i64 h.nside in
  let around = Int64.mul 4L ns in
  let i = Nx.sub_s (Nx.sub (Nx.mul_s (f1 c.face) ns) (Nx.add c.x c.y)) 1L in
  let north = Nx.less_s i ns and south = Nx.greater_s i (Int64.mul 3L ns) in
  let cap = Nx.logical_or north south in
  let n =
    Nx.where north i (Nx.where south (Nx.rsub_s around i) (Nx.full_like i ns))
  in
  let shift = Nx.where cap (Nx.zeros_like i) (land_ (Nx.sub_s i ns) 1L) in
  let j =
    Nx.div_s
      (Nx.add
         (Nx.add (Nx.mul (f2 c.face) n) (Nx.sub c.x c.y))
         (Nx.add_s shift 1L))
      2L
  in
  let j =
    Nx.where (Nx.greater_s j around) (Nx.sub_s j around)
      (Nx.where (Nx.less_s j 1L) (Nx.add_s j around) j)
  in
  let first =
    Nx.where north
      (Nx.mul_s (Nx.mul n (Nx.sub_s n 1L)) 2L)
      (Nx.where south
         (Nx.rsub_s (i64 (pixels h)) (Nx.mul_s (Nx.mul n (Nx.add_s n 1L)) 2L))
         (Nx.add_s
            (Nx.mul_s (Nx.sub_s i ns) around)
            (Int64.mul (Int64.mul 2L ns) (Int64.pred ns))))
  in
  Nx.add first (Nx.sub_s j 1L)

(* The inverse: index [p]'s ring and place, and the face that the place's
   two edge lines bound. *)
let of_ring h p =
  let ns = h.nside and npix = pixels h in
  let ncap = 2 * ns * (ns - 1) in
  let north = Nx.less_s p (i64 ncap)
  and south = Nx.greater_equal_s p (i64 (npix - ncap)) in
  (* The ring and place of the pixel [q] from a pole, counted from 0. *)
  let from_pole q =
    let i = Nx.rshift (Nx.add_s (isqrt (Nx.add_s (Nx.mul_s q 2L) 1L)) 1L) 1 in
    (i, Nx.sub (Nx.add_s q 1L) (Nx.mul_s (Nx.mul i (Nx.sub_s i 1L)) 2L))
  in
  let ni, nj = from_pole p in
  let si, sj = from_pole (Nx.rsub_s (i64 (npix - 1)) p) in
  let sj = Nx.rsub_s 1L (Nx.sub sj (Nx.mul_s si 4L)) in
  let e = Nx.sub_s p (i64 ncap) in
  let ei = Nx.add_s (Nx.rshift e (h.order + 2)) (i64 ns) in
  let ej = Nx.add_s (land_ e (i64 ((4 * ns) - 1))) 1L in
  let edge r =
    Nx.rshift (Nx.add_s (Nx.sub ej (Nx.rshift r 1)) (i64 (ns - 1))) h.order
  in
  let ire = Nx.add_s (Nx.sub_s ei (i64 ns)) 1L in
  let eface = face_of (edge (Nx.rsub_s (i64 ((2 * ns) + 2)) ire)) (edge ire) in
  let i = Nx.where north ni (Nx.where south (Nx.rsub_s (i64 (4 * ns)) si) ei) in
  let j = Nx.where north nj (Nx.where south sj ej) in
  let n = Nx.where north ni (Nx.where south si (Nx.full_like p (i64 ns))) in
  let face =
    Nx.where north
      (Nx.div (Nx.sub_s nj 1L) ni)
      (Nx.where south (Nx.add_s (Nx.div (Nx.sub_s sj 1L) si) 8L) eface)
  in
  let shift =
    Nx.where
      (Nx.logical_or north south)
      (Nx.zeros_like p)
      (land_ (Nx.add_s ei (i64 ns)) 1L)
  in
  let rt = Nx.add_s (Nx.sub i (Nx.mul_s (f1 face) (i64 ns))) 1L in
  let pt =
    Nx.sub (Nx.sub_s (Nx.mul_s j 2L) 1L) (Nx.add (Nx.mul (f2 face) n) shift)
  in
  let pt =
    Nx.where
      (Nx.greater_equal_s pt (i64 (2 * ns)))
      (Nx.sub_s pt (i64 (8 * ns)))
      pt
  in
  {
    face;
    x = Nx.rshift (Nx.sub pt rt) 1;
    y = Nx.rshift (Nx.neg (Nx.add pt rt)) 1;
  }

let to_index h c =
  match h.scheme with Ring -> to_ring h c | Nested -> to_nested h c

let of_index h p =
  match h.scheme with Ring -> of_ring h p | Nested -> of_nested h p

(* Indices are in [-1, N): -1 names no pixel, and every function here gives
   it back as no pixel. [named fn h p] is [p] with -1 read as 0, and where
   it was -1; it raises where [p] is outside [-1, N). *)
let named fn h p =
  let ok =
    Nx.logical_and (Nx.greater_equal_s p (-1L)) (Nx.less_s p (i64 (pixels h)))
  in
  Nx.check Nx.Ptree.unit ok () (fun i () ->
      Invalid_argument
        (strf "Healpix.%s: the pixel%s is outside [-1, %d)" fn (Direction.at i)
           (pixels h)));
  let none = Nx.equal_s p (-1L) in
  (Nx.where none (Nx.zeros_like p) p, none)

let minus_one_where none x = Nx.where none (Nx.full_like x (-1L)) x

let renumber h scheme p =
  let q, none = named "renumber" h p in
  if scheme = h.scheme then p
  else minus_one_where none (to_index { h with scheme } (of_index h q))

(* Directions to pixels *)

let two_thirds = 2. /. 3.

(* [place h z s tt] is the cell holding the direction at [z = cos θ], [s =
   sin θ] and longitude [tt π / 2], [tt] in [0, 4). Between the caps the
   edges are the lines of constant [1/2 + tt ∓ 3 z / 4], eq. (22), whose
   integer parts name the face and whose fractions place the direction on
   it. In a cap they are the curves of constant [tp w] and [(1 - tp) w], eqs.
   (19) and (20), [tp] the longitude within the quadrant and [w = √(3 (1 -
   |z|))], read as [s / √((1 + |z|) / 3)] to keep its precision at the
   pole. *)
let place h z s tt =
  let ns = float_of_int h.nside and last = i64 (h.nside - 1) in
  let p = Nx.sub (Nx.add_s tt 0.5) (Nx.mul_s z 0.75)
  and m = Nx.add (Nx.add_s tt 0.5) (Nx.mul_s z 0.75) in
  let fp = Nx.floor p and fm = Nx.floor m in
  let eface = face_of (Nx.cast Nx.int64 fp) (Nx.cast Nx.int64 fm) in
  let eu = Nx.sub m fm and ev = Nx.sub (Nx.add_s fp 1.) p in
  let za = Nx.abs z in
  let quadrant = Nx.minimum_s (Nx.floor tt) 3. in
  let tp = Nx.sub tt quadrant in
  let w = Nx.div s (Nx.sqrt (Nx.div_s (Nx.add_s za 1.) 3.)) in
  let cp = Nx.minimum_s (Nx.mul tp w) 1.
  and cm = Nx.minimum_s (Nx.mul (Nx.rsub_s 1. tp) w) 1. in
  let north = Nx.greater_equal_s z 0. in
  let q = Nx.cast Nx.int64 quadrant in
  let cface = Nx.where north q (Nx.add_s q 8L) in
  let cu = Nx.where north (Nx.rsub_s 1. cm) cp
  and cv = Nx.where north (Nx.rsub_s 1. cp) cm in
  let between = Nx.less_equal_s za two_thirds in
  let discrete u = Nx.minimum_s (floor_int (Nx.mul_s u ns)) last in
  {
    face = Nx.where between eface cface;
    x = discrete (Nx.where between eu cu);
    y = discrete (Nx.where between ev cv);
  }

let pixel h (d : (_, _) Direction.t) =
  let v = Direction.rows "Healpix.pixel: the vector" d.xyz in
  let x, y, z = Direction.components v in
  let rho = Nx.sqrt (Nx.add (Nx.square x) (Nx.square y)) in
  let r = Nx.sqrt (Nx.add (Nx.square rho) (Nx.square z)) in
  let z = Nx.div z r and s = Nx.div rho r in
  let lon = Quantity.value Unit.radian (Direction.lon d) in
  let tt = Nx.mul_s lon (2. /. Float.pi) in
  let tt = Nx.where (Nx.greater_equal_s tt 4.) (Nx.sub_s tt 4.) tt in
  (* A zero or NaN row reads as NaN here and names no pixel. *)
  let none = Nx.isnan z in
  let zero x = Nx.where none (Nx.zeros_like x) x in
  minus_one_where none (to_index h (place h (zero z) (zero s) (zero tt)))

(* Pixels to directions *)

(* [direction h face u v] is the point at [(u, v)] on [face]. Its ring in
   units of nside is [j = F1 - u - v]; in a cap, [n] = [j] or [4 - j] rings
   from the pole, [1 - |z| = n² / 3], eq. (4); between the caps [z = (2 - j)
   2/3], eq. (8). Its longitude is [(F2 + (u - v) / n) π / 4], [n = 1]
   between the caps, eqs. (5), (9) and (18), and 0 at a pole. *)
let direction h face u v =
  let j = Nx.sub (Nx.sub (floats (f1 face)) u) v in
  let north = Nx.less_s j 1. and south = Nx.greater_s j 3. in
  let n = Nx.where north j (Nx.where south (Nx.rsub_s 4. j) (Nx.ones_like j)) in
  let t = Nx.div_s (Nx.square n) 3. in
  let z =
    Nx.where north (Nx.rsub_s 1. t)
      (Nx.where south (Nx.sub_s t 1.) (Nx.mul_s (Nx.rsub_s 2. j) two_thirds))
  in
  let s =
    Nx.where
      (Nx.logical_or north south)
      (Nx.sqrt (Nx.mul t (Nx.rsub_s 2. t)))
      (Nx.sqrt (Nx.mul (Nx.rsub_s 1. z) (Nx.add_s z 1.)))
  in
  let pole = Nx.equal_s n 0. in
  let n = Nx.where pole (Nx.ones_like n) n in
  let phi =
    Nx.mul_s (Nx.add (floats (f2 face)) (Nx.div (Nx.sub u v) n)) (Float.pi /. 4.)
  in
  let phi = Nx.where pole (Nx.zeros_like phi) phi in
  let xyz =
    Nx.stack ~axis:(-1) [ Nx.mul s (Nx.cos phi); Nx.mul s (Nx.sin phi); z ]
  in
  ({ frame = h.frame; xyz } : (_, _) Direction.t)

(* [nowhere none d] is [d] with NaN rows where [none] holds. *)
let nowhere none (d : (_, _) Direction.t) =
  let none = Nx.unsqueeze ~axes:[ Nx.ndim none ] none in
  { d with xyz = Nx.where none (Nx.full_like d.xyz Float.nan) d.xyz }

let centre h p =
  let p, none = named "centre" h p in
  let c = of_index h p in
  let at k = Nx.div_s (Nx.add_s (floats k) 0.5) (float_of_int h.nside) in
  nowhere none (direction h c.face (at c.x) (at c.y))

(* The corners north, west, south and east: [(x + 1, y + 1)], [(x, y + 1)],
   [(x, y)] and [(x + 1, y)]. *)
let corners h p =
  let p, none = named "corners" h p in
  let c = of_index h p in
  let ns = float_of_int h.nside in
  let last k = Nx.unsqueeze ~axes:[ Nx.ndim k ] k in
  let steps a = Nx.place (Nx.beside p) (Nx.create Nx.float64 [| 4 |] a) in
  let u = Nx.div_s (Nx.add (floats (last c.x)) (steps [| 1.; 0.; 0.; 1. |])) ns
  and v =
    Nx.div_s (Nx.add (floats (last c.y)) (steps [| 1.; 1.; 0.; 0. |])) ns
  in
  nowhere (last none) (direction h (last c.face) u v)

(* Neighbours *)

(* The eight neighbours SW, W, NW, N, NE, E, SE and S as steps of [x] and
   [y]. *)
let steps_x = [| -1L; -1L; 0L; 1L; 1L; 1L; 0L; -1L |]
let steps_y = [| 0L; 1L; 1L; 1L; 0L; -1L; -1L; -1L |]

(* The base faces' adjacency, a geometric fact, laid out as HEALPix numbers
   it: the same cells as the HEALPix library's neighbour tables.

   A step off a face lands on the face across the edge or corner it
   crosses, [(ox, oy)] in [{-1, 0, 1}²], at a place read through a flip of
   [x] (1), of [y] (2), then a swap of the two (4). Four faces meet at the
   poles and at the corners on the equator; three at the corners at |z| =
   2/3, where the step east or west from a cap face's corner pixel, or north
   or south from an equatorial one's, reaches no pixel. A row's faces are
   its first plus the column [c], eastward. *)
let across row c (ox, oy) =
  let north k = Some (k mod 4)
  and eq k = Some (4 + (k mod 4))
  and south k = Some (8 + (k mod 4)) in
  match (row, ox, oy) with
  | 0, 0, 0 -> (north c, 0)
  | 0, -1, -1 -> (south c, 0)
  | 0, 0, -1 -> (eq (c + 1), 0)
  | 0, -1, 0 -> (eq c, 0)
  | 0, 1, 0 -> (north (c + 1), 5)
  | 0, 0, 1 -> (north (c + 3), 6)
  | 0, 1, 1 -> (north (c + 2), 3)
  | 1, 0, 0 -> (eq c, 0)
  | 1, 0, -1 -> (south c, 0)
  | 1, 1, -1 -> (eq (c + 1), 0)
  | 1, -1, 0 -> (south (c + 3), 0)
  | 1, 1, 0 -> (north c, 0)
  | 1, -1, 1 -> (eq (c + 3), 0)
  | 1, 0, 1 -> (north (c + 3), 0)
  | 2, 0, 0 -> (south c, 0)
  | 2, -1, -1 -> (south (c + 2), 3)
  | 2, 0, -1 -> (south (c + 1), 6)
  | 2, -1, 0 -> (south (c + 3), 5)
  | 2, 1, 0 -> (eq (c + 1), 0)
  | 2, 0, 1 -> (eq c, 0)
  | 2, 1, 1 -> (north c, 0)
  | _ -> (None, 0)

(* [across] tabulated by [k = 4 + ox + 3 oy]: the face [faces.(12 k + f)],
   -1 for none, and the move [moves.(3 k + row)]. *)
let faces, moves =
  let faces = Array.make (9 * 12) (-1L) and moves = Array.make (9 * 3) 0L in
  for k = 0 to 8 do
    let o = ((k mod 3) - 1, (k / 3) - 1) in
    for f = 0 to 11 do
      let face, move = across (f / 4) (f mod 4) o in
      Option.iter (fun g -> faces.((12 * k) + f) <- i64 g) face;
      moves.((3 * k) + (f / 4)) <- i64 move
    done
  done;
  (faces, moves)

let neighbours h p =
  let p, none = named "neighbours" h p in
  let c = of_index h p in
  let ns = i64 h.nside in
  let table a =
    Nx.place (Nx.beside p) (Nx.create Nx.int64 [| Array.length a |] a)
  in
  let last k = Nx.unsqueeze ~axes:[ Nx.ndim k ] k in
  let x = Nx.add (last c.x) (table steps_x)
  and y = Nx.add (last c.y) (table steps_y) in
  let off k =
    Nx.where (Nx.less_s k 0L) (Nx.full_like k (-1L))
      (Nx.where (Nx.greater_equal_s k ns) (Nx.ones_like k) (Nx.zeros_like k))
  in
  let ox = off x and oy = off y in
  let x = Nx.sub x (Nx.mul_s ox ns) and y = Nx.sub y (Nx.mul_s oy ns) in
  let k = Nx.add (Nx.add_s ox 4L) (Nx.mul_s oy 3L) in
  let face = last c.face in
  let face' = Nx.take (Nx.add (Nx.mul_s k 12L) face) (table faces) in
  let move = Nx.take (Nx.add (Nx.mul_s k 3L) (row face)) (table moves) in
  let has bit = Nx.not_equal_s (land_ move bit) 0L in
  let flip bit v = Nx.where (has bit) (Nx.rsub_s (Int64.pred ns) v) v in
  let x = flip 1L x and y = flip 2L y in
  let swap = has 4L in
  let missing = Nx.logical_or (Nx.less_s face' 0L) (last none) in
  let face' = Nx.where missing (Nx.zeros_like face') face' in
  minus_one_where missing
    (to_index h { face = face'; x = Nx.where swap y x; y = Nx.where swap x y })

(* Spherical harmonics *)

(* Ring [r] of the northern hemisphere, [r] in [1, 2 nside]: its [z] and [s
   = sin θ], its [n] pixels from longitude [phi0] in steps of [2π / n], the
   first at index [first] in the ring numbering. Its southern mirror, ring [4
   nside - r], has the same longitudes and starts at [12 nside² - first -
   n]. *)
type ring = { z : float; s : float; n : int; phi0 : float; first : int }

let ring h r =
  let ns = h.nside in
  let fns = float_of_int ns in
  if r < ns then
    let t = Float.pow (float_of_int r /. fns) 2. /. 3. in
    {
      z = 1. -. t;
      s = Float.sqrt (t *. (2. -. t));
      n = 4 * r;
      phi0 = Float.pi /. float_of_int (4 * r);
      first = 2 * r * (r - 1);
    }
  else
    let z = float_of_int ((2 * ns) - r) *. two_thirds /. fns in
    {
      z;
      s = Float.sqrt ((1. -. z) *. (1. +. z));
      n = 4 * ns;
      phi0 = (if (r - ns) land 1 = 0 then Float.pi /. (4. *. fns) else 0.);
      first = (2 * ns * (ns - 1)) + ((r - ns) * 4 * ns);
    }

(* Rings of one length, transformed together: the caps' rings one by one,
   then the [nside + 1] rings from [nside] to the equator. [index] holds
   their pixels in [h]'s numbering, [[2; k; n]], the northern rings then
   their mirrors, -1 for the equator's. *)
type group = { rings : ring array; index : (int64, Nx.int64_elt, Nx.host) Nx.t }

let groups h =
  let ns = h.nside and npix = pixels h in
  let group rs =
    let rings = Array.map (ring h) rs in
    let n = rings.(0).n and k = Array.length rs in
    let column f = Nx.create Nx.int64 [| k; 1 |] (Array.map f rings) in
    let along = Nx.reshape [| 1; n |] (Nx.arange Nx.int64 0 n 1) in
    let north = Nx.add (column (fun g -> i64 g.first)) along in
    let south = Nx.add (column (fun g -> i64 (npix - g.first - n))) along in
    let mirrored =
      Nx.create Nx.bool [| k; 1 |] (Array.map (fun r -> r < 2 * ns) rs)
    in
    let south = Nx.where mirrored south (Nx.full_like south (-1L)) in
    let index = Nx.stack ~axis:0 [ north; south ] in
    let index =
      match h.scheme with
      | Ring -> index
      | Nested ->
          minus_one_where (Nx.less_s index 0L)
            (to_nested h (of_ring h (Nx.maximum_s index 0L)))
    in
    { rings; index }
  in
  let caps = List.init (ns - 1) (fun i -> group [| i + 1 |]) in
  caps @ [ group (Array.init (ns + 1) (fun i -> ns + i)) ]

(* [gather h groups] is, for each pixel in [h]'s numbering, its position in
   the groups' indices laid end to end; the equator's -1 writes nothing. *)
let gather h groups =
  let order =
    Nx.concatenate ~axis:0 (List.map (fun g -> Nx.flatten g.index) groups)
  in
  let positions = Nx.arange Nx.int64 0 (Nx.numel order) 1 in
  Nx.scatter ~axis:0 order positions (Nx.zeros Nx.int64 [| pixels h |])

let rings_of groups = Array.concat (List.map (fun g -> g.rings) groups)

(* The Legendre functions

   λ_lm(cos θ), orthonormal with the Condon–Shortley phase, follows the
   recurrence in [l] that is stable upward (Schaeffer 2013, §3.2):

   λ_00 = 1 / √(4π),   λ_mm = -√((2m + 1) / (2m)) sin θ λ_(m-1)(m-1),
   λ_lm = a_lm cos θ λ_(l-1)m - b_lm λ_(l-2)m,
   a_lm = √((4l² - 1) / (l² - m²)),   b_lm = a_lm / a_(l-1)m,

   with [b_(m+1)m = 0]. λ_mm is of order sin^m θ, which leaves float64's
   range near the poles while the λ_lm it starts grow back into it, so each
   value is held with a count [k] of factors 2^-600, an exponent that λ_mm
   decrements when it falls below 2^-600 and the recurrence increments when
   it rises above 2^600 (Schaeffer 2013, §3.5). A value at [k ≤ -2] is
   below 2^-600 and reads as 0.

   The λ_lm stream through one [Rune.scan] of [lmax + 2] steps, so a
   compiled transform's program does not grow with [lmax]. Its state is one
   value per ring and {e pair} of orders [p] and [lmax - p]: order [p] for
   [lmax + 1 - p] steps, [l] from [p] to [lmax], then order [lmax - p] for
   [p + 1] steps. Each step computes one degree of every pair, and none past
   [lmax]. *)

let big = 0x1p600
let small = 0x1p-600

(* The loop's plan, host tables of [steps × pairs] cells: [a] and [b];
   [start] where an order starts, at its λ_mm; [second] where the pair is at
   its second order; the cell's degree [l] and order [m], -1 once the pair
   is done. [diagonal] is each ring's λ_mm and [exponent] its count, [rings ×
   (lmax + 1)]. *)
type plan = {
  steps : int;
  pairs : int;
  a : float array;
  b : float array;
  start : float array;
  second : float array;
  degree : int array;
  order : int array;
  diagonal : float array;
  exponent : float array;
}

let plan rings lmax =
  let l1 = lmax + 1 in
  let pairs = (l1 + 1) / 2 and steps = l1 + 1 in
  let cells v = Array.make (steps * pairs) v in
  let a = cells 0. and b = cells 0. and start = cells 0. in
  let second = cells 0. and degree = cells (-1) and order = cells (-1) in
  let coefficient l m =
    let fl = float_of_int l and fm = float_of_int m in
    Float.sqrt (((4. *. fl *. fl) -. 1.) /. ((fl *. fl) -. (fm *. fm)))
  in
  for p = 0 to pairs - 1 do
    let orders =
      if lmax - p = p then [ (p, 0) ] else [ (p, 0); (lmax - p, l1 - p) ]
    in
    List.iter
      (fun (m, from) ->
        for l = m to lmax do
          let i = ((from + l - m) * pairs) + p in
          degree.(i) <- l;
          order.(i) <- m;
          if from > 0 then second.(i) <- 1.;
          if l = m then start.(i) <- 1.
          else begin
            a.(i) <- coefficient l m;
            if l > m + 1 then b.(i) <- coefficient l m /. coefficient (l - 1) m
          end
        done)
      orders
  done;
  let nr = Array.length rings in
  let diagonal = Array.make (nr * l1) 0.
  and exponent = Array.make (nr * l1) 0. in
  Array.iteri
    (fun r g ->
      let d = ref (1. /. Float.sqrt (4. *. Float.pi)) and k = ref 0. in
      for m = 0 to lmax do
        if m > 0 then begin
          let fm = float_of_int m in
          d := -. !d *. g.s *. Float.sqrt (((2. *. fm) +. 1.) /. (2. *. fm));
          if Float.abs !d < small then begin
            d := !d *. big;
            k := !k -. 1.
          end
        end;
        diagonal.((r * l1) + m) <- !d;
        exponent.((r * l1) + m) <- !k
      done)
    rings;
  { steps; pairs; a; b; start; second; degree; order; diagonal; exponent }

(* [cells s ~none f] is [f l m] at each cell, [none] where the pair is
   done. *)
let cells s ~none f =
  Array.init (s.steps * s.pairs) (fun i ->
      if s.degree.(i) < 0 then none else f s.degree.(i) s.order.(i))

let host like shape a = Nx.place (Nx.beside like) (Nx.create Nx.float64 shape a)

let positions like shape a =
  Nx.place (Nx.beside like) (Nx.create Nx.int64 shape (Array.map i64 a))

(* [legendre like rings s c y f ~init xs] runs the loop beside [like]: [f
   carry ~second rows lambda] is a step's use of λ, [[rings; pairs]], with
   [second] the step's row of [s.second] and [rows] its rows of [xs]; the
   result is the final carry and the outputs, as [Rune.scan]'s. *)
let legendre like rings s c y f ~init xs =
  let nr = Array.length rings
  and l1 = Array.length s.diagonal / Array.length rings in
  let cos = host like [| nr; 1 |] (Array.map (fun g -> g.z) rings) in
  (* A pair's values at its first and second order's start, [[rings;
     pairs]]. *)
  let starts values =
    let at pick =
      host like [| nr; s.pairs |]
        (Array.init (nr * s.pairs) (fun i ->
             values.((i / s.pairs * l1) + pick (i mod s.pairs))))
    in
    (at Fun.id, at (fun p -> l1 - 1 - p))
  in
  let d1, d2 = starts s.diagonal and k1, k2 = starts s.exponent in
  let table a = host like [| s.steps; s.pairs |] a in
  let zeros = Nx.zeros_like d1 in
  let step (state, carry) rows =
    match (state, rows) with
    | [ v1; v2; k ], a :: b :: start :: second :: rest ->
        let v = Nx.sub (Nx.mul a (Nx.mul cos v1)) (Nx.mul b v2) in
        let starting = Nx.greater_s start 0.
        and late = Nx.greater_s second 0. in
        let v = Nx.where starting (Nx.where late d2 d1) v
        and k = Nx.where starting (Nx.where late k2 k1) k in
        (* Only a value at [k < 0] exceeds 2^600: at [k = 0] it is a λ_lm,
           below √((2l + 1) / 4π). *)
        let up = Nx.greater_s (Nx.abs v) big in
        let v = Nx.where up (Nx.mul_s v small) v
        and previous = Nx.where up (Nx.mul_s v1 small) v1
        and k = Nx.where up (Nx.add_s k 1.) k in
        let zero = Nx.scalar_like v 0. in
        let lambda =
          Nx.where (Nx.less_s k (-0.5))
            (Nx.where (Nx.less_s k (-1.5)) zero (Nx.mul_s v small))
            v
        in
        let carry, out = f carry ~second rest lambda in
        (([ v; previous; k ], carry), out)
    | _ -> assert false
  in
  let open Nx.Ptree in
  let (_, carry), out =
    Rune.scan
      (pair (list tensor) c)
      (list tensor) y step
      ~init:([ zeros; zeros; zeros ], init)
      ([ table s.a; table s.b; table s.start; table s.second ] @ xs)
  in
  (carry, out)

let parts z = (Nx.real Nx.float64 z, Nx.imag Nx.float64 z)
let complex re im = Nx.complex Nx.complex128 ~re ~im

let take_axis ~axis i x =
  Nx.slice (List.init axis (fun _ -> Nx.A) @ [ Nx.I i ]) x

(* [phases like rings lmax sign w] is [w e^(sign i m φ0)], [[rings; lmax +
   1]]. *)
let phases like rings lmax sign w =
  let k = Array.length rings in
  let part f =
    Nx.place (Nx.beside like)
      (Nx.init Nx.float64
         [| k; lmax + 1 |]
         (fun ix -> w *. f (sign *. float_of_int ix.(1) *. rings.(ix.(0)).phi0)))
  in
  complex (part Float.cos) (part Float.sin)

(* [analyse h groups lmax map] is the quadrature
   a_lm = (4π / N) Σ_p f_p λ_lm(θ_p) e^(-i m φ_p) over the [N] pixels: each
   ring's Fourier series by an FFT of its [n] pixels, its coefficient [m]
   the FFT's bin [m mod n], then the sums over rings, each mirror ring
   folded in by λ_lm(-z) = (-1)^(l+m) λ_lm(z). *)
let analyse h groups lmax map =
  let on = Nx.beside map and l1 = lmax + 1 in
  let w = 4. *. Float.pi /. float_of_int (pixels h) in
  let batch = Nx.ndim map - 1 in
  let group g =
    let n = g.rings.(0).n in
    let block = Nx.take ~axis:batch (Nx.place on g.index) map in
    let axis = Nx.ndim block - 1 in
    let spectrum =
      Nx.fft ~axes:[ axis ] (complex block (Nx.zeros_like block))
    in
    let bins = positions map [| l1 |] (Array.init l1 (fun m -> m mod n)) in
    Nx.mul (Nx.take ~axis bins spectrum) (phases map g.rings lmax (-1.) w)
  in
  let g = Nx.concatenate ~axis:(batch + 1) (List.map group groups) in
  let north = take_axis ~axis:batch 0 g and south = take_axis ~axis:batch 1 g in
  let rings = rings_of groups in
  let s = plan rings lmax in
  (* Each pair's even and odd sums at its first and second order, real and
     imaginary, [[...; rings; pairs]]. *)
  let at pick =
    let columns = positions map [| s.pairs |] (Array.init s.pairs pick) in
    List.concat_map
      (fun x ->
        let re, im = parts (Nx.take ~axis:(batch + 1) columns x) in
        [ re; im ])
      [ Nx.add north south; Nx.sub north south ]
  in
  let first = at Fun.id and second = at (fun p -> lmax - p) in
  let even =
    host map [| s.steps; s.pairs |]
      (cells s ~none:0. (fun l m -> if (l + m) land 1 = 0 then 1. else 0.))
  in
  let f () ~second:late rows lambda =
    match (rows, first, second) with
    | [ even ], [ e1r; e1i; o1r; o1i ], [ e2r; e2i; o2r; o2i ] ->
        let late = Nx.greater_s late 0. and even = Nx.greater_s even 0. in
        let pick e1 o1 e2 o2 =
          Nx.where late (Nx.where even e2 o2) (Nx.where even e1 o1)
        in
        let sum x = Nx.sum ~axes:[ Nx.ndim x - 2 ] (Nx.mul lambda x) in
        ((), [ sum (pick e1r o1r e2r o2r); sum (pick e1i o1i e2i o2i) ])
    | _ -> assert false
  in
  let (), out =
    legendre map rings s Nx.Ptree.unit
      Nx.Ptree.(list tensor)
      f ~init:() [ even ]
  in
  (* The steps' outputs, [[steps; ...; pairs]], read at each [[l; m]]. *)
  let cell = Array.make (l1 * l1) (-1) in
  Array.iteri
    (fun i l -> if l >= 0 then cell.((l * l1) + s.order.(i)) <- i)
    s.degree;
  let read y =
    let y = Nx.moveaxis 0 batch y in
    let shape = Array.sub (Nx.shape y) 0 batch in
    let y = Nx.reshape (Array.append shape [| s.steps * s.pairs |]) y in
    Nx.take ~axis:batch (positions map [| l1; l1 |] cell) y
  in
  match out with
  | [ re; im ] ->
      (* A real map's coefficients at m = 0 are real. *)
      let real =
        Nx.place on
          (Nx.create Nx.bool [| l1 |] (Array.init l1 (fun m -> m = 0)))
      in
      let im = read im in
      complex (read re) (Nx.where real (Nx.zeros_like im) im)
  | _ -> assert false

(* [synthesise groups inverse alm] is f_p = Σ_lm a_lm λ_lm(θ_p) e^(i m φ_p)
   at each pixel, over [m] from [-lmax] to [lmax] with a_l(-m) = (-1)^m conj
   a_lm: each ring's coefficients F_m summed over [l], those of its mirror
   by λ_lm(-z) = (-1)^(l+m) λ_lm(z), then folded onto the ring's [n] points
   by [m mod n] and summed by an inverse FFT. *)
let synthesise groups inverse alm =
  let on = Nx.beside alm in
  let lmax = (Nx.shape alm).(Nx.ndim alm - 1) - 1 in
  let l1 = lmax + 1 and batch = Nx.ndim alm - 2 in
  let leading = Array.sub (Nx.shape alm) 0 batch in
  let rings = rings_of groups in
  let s = plan rings lmax in
  (* Each step's coefficients, [[steps; ...; 1; pairs]]. *)
  let rows x =
    let cell = cells s ~none:(-1) (fun l m -> (l * l1) + m) in
    let flat = Nx.reshape (Array.append leading [| l1 * l1 |]) x in
    let y =
      Nx.take ~axis:batch (positions alm [| s.steps; s.pairs |] cell) flat
    in
    Nx.moveaxis batch 0 (Nx.unsqueeze ~axes:[ batch + 1 ] y)
  in
  let re, im = parts alm in
  let sign =
    host alm [| s.steps; s.pairs |]
      (cells s ~none:0. (fun l m -> if (l + m) land 1 = 0 then 1. else -1.))
  in
  let zeros =
    Nx.place on
      (Nx.zeros Nx.float64
         (Array.append leading [| Array.length rings; s.pairs |]))
  in
  (* The northern and southern sums of each pair's first and second order,
     real and imaginary. *)
  let f sums ~second rows lambda =
    match (sums, rows) with
    | [ n1r; n1i; s1r; s1i; n2r; n2i; s2r; s2i ], [ sign; ar; ai ] ->
        let l2 = Nx.mul lambda second in
        let l1 = Nx.sub lambda l2 in
        let m1 = Nx.mul l1 sign and m2 = Nx.mul l2 sign in
        let add acc l a = Nx.add acc (Nx.mul l a) in
        ( [
            add n1r l1 ar;
            add n1i l1 ai;
            add s1r m1 ar;
            add s1i m1 ai;
            add n2r l2 ar;
            add n2i l2 ai;
            add s2r m2 ar;
            add s2i m2 ai;
          ],
          () )
    | _ -> assert false
  in
  let sums, () =
    legendre alm rings s
      Nx.Ptree.(list tensor)
      Nx.Ptree.unit f
      ~init:(List.init 8 (fun _ -> zeros))
      [ sign; rows re; rows im ]
  in
  (* Columns [m] of the pairs' sums, [[...; rings; lmax + 1]]. *)
  let columns first second =
    let both = Nx.concatenate ~axis:(batch + 1) [ first; second ] in
    let at =
      Array.init l1 (fun m -> if m < s.pairs then m else s.pairs + lmax - m)
    in
    Nx.take ~axis:(batch + 1) (positions alm [| l1 |] at) both
  in
  let north, south =
    match sums with
    | [ n1r; n1i; s1r; s1i; n2r; n2i; s2r; s2i ] ->
        ( complex (columns n1r n2r) (columns n1i n2i),
          complex (columns s1r s2r) (columns s1i s2i) )
    | _ -> assert false
  in
  let f = Nx.stack ~axis:batch [ north; south ] in
  let start = ref 0 in
  let group g =
    let n = g.rings.(0).n and k = Array.length g.rings in
    let rows =
      Nx.slice
        (List.init (batch + 1) (fun _ -> Nx.A) @ [ Nx.R (!start, !start + k) ])
        f
    in
    start := !start + k;
    let positive = Nx.mul rows (phases alm g.rings lmax 1. 1.) in
    let last = Nx.ndim positive - 1 in
    let negative =
      Nx.conjugate
        (Nx.slice (List.init last (fun _ -> Nx.A) @ [ Nx.R (1, l1) ]) positive)
    in
    let terms = Nx.concatenate ~axis:last [ positive; negative ] in
    let bins =
      Array.init
        ((2 * lmax) + 1)
        (fun i ->
          let m = if i <= lmax then i else lmax - i in
          ((m mod n) + n) mod n)
    in
    let bins =
      Nx.broadcast_to (Nx.shape terms) (positions alm [| (2 * lmax) + 1 |] bins)
    in
    let shape = Array.copy (Nx.shape terms) in
    shape.(last) <- n;
    let folded =
      Nx.scatter ~combine:Nx.Add ~axis:last bins terms
        (Nx.place on (Nx.zeros Nx.complex128 shape))
    in
    let values =
      Nx.real Nx.float64 (Nx.ifft ~axes:[ last ] ~norm:Nx.Forward folded)
    in
    Nx.reshape (Array.append leading [| 2 * k * n |]) values
  in
  let flat = Nx.concatenate ~axis:batch (List.map group groups) in
  Nx.take ~axis:batch (Nx.place on inverse) flat

let shape_text x =
  String.concat "; " (Array.to_list (Array.map string_of_int (Nx.shape x)))

let analysis h ~lmax ~iterations map =
  let n = Nx.ndim map in
  if n = 0 || (Nx.shape map).(n - 1) <> pixels h then
    invalid_arg
      (strf
         "Healpix.analysis: expected maps of %d pixels on the last axis, got \
          shape [%s]"
         (pixels h) (shape_text map));
  if lmax < 0 then
    invalid_arg (strf "Healpix.analysis: lmax %d is negative" lmax);
  if iterations < 0 then
    invalid_arg (strf "Healpix.analysis: iterations %d is negative" iterations);
  let groups = groups h in
  let inverse = gather h groups in
  let alm = ref (analyse h groups lmax map) in
  for _ = 1 to iterations do
    let residual = Nx.sub map (synthesise groups inverse !alm) in
    alm := Nx.add !alm (analyse h groups lmax residual)
  done;
  !alm

let synthesis h alm =
  let shape = Nx.shape alm in
  let n = Array.length shape in
  if n < 2 || shape.(n - 1) <> shape.(n - 2) || shape.(n - 1) = 0 then
    invalid_arg
      (strf
         "Healpix.synthesis: expected coefficients [...; lmax + 1; lmax + 1], \
          got shape [%s]"
         (shape_text alm));
  let groups = groups h in
  synthesise groups (gather h groups) alm
