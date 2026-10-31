(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* H-compress decoding (FITS 4.0 §10.4.4, White 1992), as cfitsio's
   fits_hdecompress64 reads a stream:

     DD 99 | nx (4) | ny (4) | scale (4) | h0 (8) | planes (3) | bits | signs

   big-endian, [ny] the fastest axis. The bits code the magnitudes of the
   coefficients, the first one excepted, in four quadrants, the second and
   third sharing a plane count. Each bit plane is a nybble, 0 for the plane
   written as is, a nybble per 2×2 block, or 15 for a quadtree of 4-bit codes
   in a fixed Huffman code. A zero nybble ends the bits, and the signs of
   the nonzero coefficients follow from the next byte, one bit each. The
   coefficients, times the scale, invert to the pixels.

   A few bytes can code a tile of any size, so decoding costs what the
   stream and the window hold, never the tile: the nonzero coefficients,
   one per sign bit, are kept sorted by position, a quadtree expands only
   where its codes flag, and the inverse transform runs only over the
   window's ancestors at each level. *)

open Err
module A = Bigarray.Array1

type ints = (int, Bigarray.int_elt, Bigarray.c_layout) A.t

let magic = (0xDD, 0x99)

(* A magnitude of this many bits is at most [max_int]. *)
let max_planes = 62

(* Bit input, most significant bit first *)

type bits = {
  src : Checksum.bigbytes;
  mutable pos : int;
  mutable byte : int;
  mutable left : int; (* bits of [byte] not yet read *)
}

let bits src = { src; pos = 0; byte = 0; left = 0 }

let byte r =
  if r.pos >= A.dim r.src then fail "the stream ends early";
  let b = A.unsafe_get r.src r.pos in
  r.pos <- r.pos + 1;
  b

let bit r =
  if r.left = 0 then begin
    r.byte <- byte r;
    r.left <- 8
  end;
  r.left <- r.left - 1;
  (r.byte lsr r.left) land 1

let nbits r n =
  let v = ref 0 in
  for _ = 1 to n do
    v := (!v lsl 1) lor bit r
  done;
  !v

(* [be r n] reads [n] whole bytes as a big-endian number. *)
let be r n =
  let v = ref 0 in
  for _ = 1 to n do
    v := (!v lsl 8) lor byte r
  done;
  !v

(* The quadtree's fixed Huffman code (White 1992): value [v] is the code
   [codes.(v)] of [lengths.(v)] bits. *)
let codes =
  [|
    0x3e; 0x00; 0x01; 0x08; 0x02; 0x09; 0x1a; 0x1b;
    0x03; 0x1c; 0x0a; 0x1d; 0x0b; 0x1e; 0x3f; 0x0c;
  |]

let lengths = [| 6; 3; 3; 4; 3; 4; 5; 5; 3; 5; 4; 5; 4; 5; 6; 4 |]

(* [values.((l lsl 6) lor c)] is the value of the code [c] of [l] bits, or
   -1. The code is complete, so every 6 bits end a code. *)
let values =
  let t = Array.make (7 lsl 6) (-1) in
  Array.iteri (fun v c -> t.((lengths.(v) lsl 6) lor c) <- v) codes;
  t

let huffman r =
  let c = ref (nbits r 3) and l = ref 3 in
  while values.((!l lsl 6) lor !c) < 0 do
    c := (!c lsl 1) lor bit r;
    incr l
  done;
  values.((!l lsl 6) lor !c)

(* The header *)

type header = {
  nx : int;
  ny : int;
  scale : int;
  h0 : int;
  planes : int array;
}

let header r =
  let m0 = byte r in
  let m1 = byte r in
  if (m0, m1) <> magic then fail "the stream does not start with DD 99";
  let nx = Rice.signed 4 (be r 4) in
  let ny = Rice.signed 4 (be r 4) in
  let scale = Rice.signed 4 (be r 4) in
  if nx < 1 || ny < 1 then fail "the stream holds %d x %d pixels" nx ny;
  let h = ref 0L in
  for _ = 1 to 8 do
    h := Int64.logor (Int64.shift_left !h 8) (Int64.of_int (byte r))
  done;
  let planes = Array.init 3 (fun _ -> byte r) in
  Array.iter
    (fun p ->
      if p > max_planes then
        fail "a quadrant has %d bit planes, past %d" p max_planes)
    planes;
  { nx; ny; scale; h0 = Int64.to_int !h; planes }

let size src =
  let h = header (bits src) in
  (h.nx, h.ny)

(* Bit planes *)

(* The least [k] with [2^k >= n]. *)
let log2_up n =
  let k = ref 0 in
  while 1 lsl !k < n do
    incr k
  done;
  !k

let half_up n = (n + 1) / 2

(* The bits the planes set: position [keys.(i)], [x * ny + y], has the bit
   [ors.(i)], for [i] below [count]. A position may repeat. *)
type marks = {
  mutable keys : int array;
  mutable ors : int array;
  mutable count : int;
}

let set_bit a k plane =
  if a.count = Array.length a.keys then begin
    let grow x = Array.append x (Array.make (Int.max 16 a.count) 0) in
    a.keys <- grow a.keys;
    a.ors <- grow a.ors
  end;
  a.keys.(a.count) <- k;
  a.ors.(a.count) <- plane;
  a.count <- a.count + 1

(* The nonzero magnitudes: [mag.(i)] at [at.(i)], positions increasing. *)
type coefs = { at : int array; mag : int array }

(* [coefs a] is the magnitudes [a]'s bits make: positions sorted once and
   their bits merged. *)
let coefs a =
  let order = Array.init a.count Fun.id in
  Array.stable_sort (fun i j -> Int.compare a.keys.(i) a.keys.(j)) order;
  let at = Array.make a.count 0 and mag = Array.make a.count 0 in
  let k = ref (-1) in
  Array.iter
    (fun i ->
      if !k >= 0 && at.(!k) = a.keys.(i) then
        mag.(!k) <- mag.(!k) lor a.ors.(i)
      else begin
        incr k;
        at.(!k) <- a.keys.(i);
        mag.(!k) <- a.ors.(i)
      end)
    order;
  { at = Array.sub at 0 (!k + 1); mag = Array.sub mag 0 (!k + 1) }

(* [find c k] is the magnitude at position [k], or 0. *)
let find c k =
  let lo = ref 0 and hi = ref (Array.length c.at) in
  while !lo < !hi do
    let mid = (!lo + !hi) / 2 in
    if c.at.(mid) < k then lo := mid + 1 else hi := mid
  done;
  if !lo < Array.length c.at && c.at.(!lo) = k then c.mag.(!lo) else 0

(* [mark a off n nqx nqy plane i j code] ors [plane] into the pixels of the
   [nqx × nqy] quadrant at [off], rows [n] apart, that the 4-bit [code] of
   its 2×2 block [(i, j)] marks: bit 3 for the block's pixel [(2i, 2j)], 2
   for [(2i, 2j + 1)], 1 for [(2i + 1, 2j)] and 0 for [(2i + 1, 2j + 1)]. *)
let mark a off n nqx nqy plane i j code =
  for di = 0 to 1 do
    for dj = 0 to 1 do
      let x = (2 * i) + di and y = (2 * j) + dj in
      if x < nqx && y < nqy && (code lsr (3 - (2 * di) - dj)) land 1 = 1 then
        set_bit a (off + (x * n) + y) plane
    done
  done

(* [level levels k q] is the size along an axis of [q] entries of level [k]
   of a quadtree of [levels]: [q] halved, rounded up, [levels - k] times, at
   least 1. *)
let level levels k q =
  let d = 1 lsl (levels - k) in
  Int.max 1 ((q + d - 1) / d)

(* [expand r codes nx ny] is the nonzero codes, each [(i, j, code)], of the
   quadtree level of [nx × ny] entries below the level whose nonzero codes
   are [codes]: a code's bits flag the four entries of its block, placed as
   {!mark} places them, and each flagged entry reads its code, the last in
   raster order first. *)
let expand r codes nx ny =
  let flagged = ref [] in
  List.iter
    (fun (i, j, code) ->
      for di = 0 to 1 do
        for dj = 0 to 1 do
          let x = (2 * i) + di and y = (2 * j) + dj in
          if x < nx && y < ny && (code lsr (3 - (2 * di) - dj)) land 1 = 1
          then flagged := ((x * ny) + y) :: !flagged
        done
      done)
    codes;
  let flagged = Array.of_list !flagged in
  Array.sort (fun a b -> Int.compare b a) flagged;
  Array.fold_left
    (fun acc k ->
      match huffman r with 0 -> acc | c -> (k / ny, k mod ny, c) :: acc)
    [] flagged

(* [quadrant r a off n nqx nqy planes] reads the [planes] bit planes of the
   [nqx × nqy] quadrant at [off], rows [n] apart, the highest first. A
   quadtree descends from one code over [log2_up (max nqx nqy)] levels to
   the quadrant's blocks. *)
let quadrant r a off n nqx nqy planes =
  let levels = log2_up (Int.max nqx nqy) in
  let w = half_up nqy in
  for bit = planes - 1 downto 0 do
    let plane = 1 lsl bit in
    match nbits r 4 with
    | 0 ->
        (* the plane as it is, a nybble per block *)
        for k = 0 to (half_up nqx * w) - 1 do
          match nbits r 4 with
          | 0 -> ()
          | code -> mark a off n nqx nqy plane (k / w) (k mod w) code
        done
    | 0xF ->
        let first = huffman r in
        let codes = ref (if first = 0 then [] else [ (0, 0, first) ]) in
        for k = 1 to levels - 1 do
          codes := expand r !codes (level levels k nqx) (level levels k nqy)
        done;
        List.iter
          (fun (i, j, code) -> mark a off n nqx nqy plane i j code)
          !codes
    | c -> fail "a bit plane's code is %d, not 0 or 15" c
  done

(* Inverse H-transform *)

(* [round v pos neg mask] rounds [v] to the multiple that [mask] keeps,
   adding [pos] to a nonnegative [v] and [neg] to a negative one. *)
let round v pos neg mask = (v + if v >= 0 then pos else neg) land mask

(* A window of a level: rows [x0, x1) and columns [y0, y1), its values row
   by row. *)
type win = { x0 : int; x1 : int; y0 : int; y1 : int; v : int array }

let win x0 x1 y0 y1 =
  { x0; x1; y0; y1; v = Array.make ((x1 - x0) * (y1 - y0)) 0 }

let get w x y = Array.unsafe_get w.v (((x - w.x0) * (w.y1 - w.y0)) + y - w.y0)

let put w x y v =
  if x >= w.x0 && x < w.x1 && y >= w.y0 && y < w.y1 then
    Array.unsafe_set w.v (((x - w.x0) * (w.y1 - w.y0)) + y - w.y0) v

(* [nudge h target shift smax] moves [h] toward [target / 2^shift], the
   change rounded toward zero and at most [smax]. *)
let nudge h target shift smax =
  let s = target - (h lsl shift) in
  let s = if s >= 0 then s asr shift else (s + (1 lsl shift) - 1) asr shift in
  h + Int.max (Int.min s smax) (-smax)

(* [slope h hm h0 hp smax] smooths the slope [h] of a block of mean [h0]
   between neighbours of means [hm] and [hp], within monotonicity. *)
let slope h hm h0 hp smax =
  let dmax = Int.max (Int.min (hp - h0) (h0 - hm)) 0 lsl 2 in
  let dmin = Int.min (Int.max (hp - h0) (h0 - hm)) 0 lsl 2 in
  if dmin < dmax then nudge h (Int.max (Int.min (hp - hm) dmax) dmin) 3 smax
  else h

(* [curvature hc hx hy h0 hmm hpm hmp hpp smax] smooths the curvature [hc]
   of a block of mean [h0] and smoothed slopes [hx], [hy], whose diagonal
   neighbours have means [hmm] to [hpp], [m] before and [p] after along
   each axis. *)
let curvature hc hx hy h0 hmm hpm hmp hpp smax =
  let hx2 = hx lsl 1 and hy2 = hy lsl 1 in
  let m1 =
    Int.min
      (Int.max (hpp - h0) 0 - hx2 - hy2)
      (Int.max (h0 - hpm) 0 + hx2 - hy2)
  and m2 =
    Int.min
      (Int.max (h0 - hmp) 0 - hx2 + hy2)
      (Int.max (hmm - h0) 0 + hx2 + hy2)
  in
  let dmax = Int.min m1 m2 lsl 4 in
  let m1 =
    Int.max
      (Int.min (hpp - h0) 0 - hx2 - hy2)
      (Int.min (h0 - hpm) 0 + hx2 - hy2)
  and m2 =
    Int.max
      (Int.min (h0 - hmp) 0 - hx2 + hy2)
      (Int.min (hmm - h0) 0 + hx2 + hy2)
  in
  let dmin = Int.max m1 m2 lsl 4 in
  if dmin < dmax then
    nudge hc (Int.max (Int.min (hpp + hmm - hmp - hpm) dmax) dmin) 6 smax
  else hc

(* [top n k] is the size along an axis of [n] pixels at level [k], 0 the
   finest: [n] halved, rounded up, [k] times. *)
let top n k =
  let d = 1 lsl k in
  (n + d - 1) / d

(* [invert coef nx ny ~smooth scale w] fills the window [w] of the
   [nx × ny] tile whose coefficients [coef] gives.

   Level [k] holds [top nx k × top ny k] values. Its 2×2 block [(2p, 2q)]
   has the mean [h0], the value [(p, q)] of the level above, and the slopes
   [hx], [hy] and curvature [hc] of the coefficients at [(px + p, q)],
   [(p, py + q)] and [(px + p, py + q)], [px × py] the level above's size.
   The coefficients round to the multiples their level's divisions left,
   the low bits the forward transform discarded pass from [hc], [hx] and
   [hy] to [h0], and the block's four sums halve, quarter on the finest
   level; a block cut by the level's last row or column has fewer.
   Smoothing moves a block's slopes and curvature by its neighbours' means,
   blocks on the level's edge excepted, so each level above the window is
   computed one block wider each way. A tile of one pixel has no level. *)
let invert coef nx ny ~smooth scale w =
  let levels = log2_up (Int.max nx ny) in
  let smax = scale asr 1 in
  let sm = smooth && smax > 0 in
  let margin = if sm then 1 else 0 in
  (* the window's ancestors, finest first *)
  let wins = Array.make (levels + 1) w in
  for k = 1 to levels do
    let b = wins.(k - 1) in
    wins.(k) <-
      win
        (Int.max 0 ((b.x0 / 2) - margin))
        (Int.min (top nx k) (((b.x1 - 1) / 2) + 1 + margin))
        (Int.max 0 ((b.y0 / 2) - margin))
        (Int.min (top ny k) (((b.y1 - 1) / 2) + 1 + margin))
  done;
  if levels = 0 then put w 0 0 (coef 0)
  else begin
    let bit2 = 1 lsl (levels + 1) in
    let prnd2 = bit2 asr 1 in
    put wins.(levels) 0 0 (round (coef 0) prnd2 (prnd2 - 1) (-bit2))
  end;
  for k = levels - 1 downto 0 do
    let o = wins.(k) and up = wins.(k + 1) in
    let nxtop = top nx k and nytop = top ny k in
    let px = top nx (k + 1) and py = top ny (k + 1) in
    let bit0 = 1 lsl k in
    let bit1 = bit0 lsl 1 in
    let prnd0 = bit0 asr 1 and prnd1 = bit0 in
    let nrnd0 = if k = 0 then 0 else prnd0 - 1 and nrnd1 = prnd1 - 1 in
    let mask0 = -bit0 and mask1 = -bit1 in
    let shift = if k = 0 then 2 else 1 in
    for p = o.x0 / 2 to (o.x1 - 1) / 2 do
      for q = o.y0 / 2 to (o.y1 - 1) / 2 do
        let i = 2 * p and j = 2 * q in
        let h0 = get up p q in
        let has_x = i + 1 < nxtop and has_y = j + 1 < nytop in
        let sx = sm && i >= 2 && i < nxtop - 2 in
        let sy = sm && j >= 2 && j < nytop - 2 in
        let hx =
          if not has_x then 0
          else
            let h = coef (((px + p) * ny) + q) in
            if sx then slope h (get up (p - 1) q) h0 (get up (p + 1) q) smax
            else h
        in
        let hy =
          if not has_y then 0
          else
            let h = coef ((p * ny) + py + q) in
            if sy then slope h (get up p (q - 1)) h0 (get up p (q + 1)) smax
            else h
        in
        if has_x && has_y then begin
          let hc = coef (((px + p) * ny) + py + q) in
          let hc =
            if sx && sy then
              curvature hc hx hy h0
                (get up (p - 1) (q - 1))
                (get up (p + 1) (q - 1))
                (get up (p - 1) (q + 1))
                (get up (p + 1) (q + 1))
                smax
            else hc
          in
          let hx = round hx prnd1 nrnd1 mask1 in
          let hy = round hy prnd1 nrnd1 mask1 in
          let hc = round hc prnd0 nrnd0 mask0 in
          let lowbit0 = hc land bit0 in
          let hx = if hx >= 0 then hx - lowbit0 else hx + lowbit0 in
          let hy = if hy >= 0 then hy - lowbit0 else hy + lowbit0 in
          let lowbit1 = (hc lxor hx lxor hy) land bit1 in
          let h0 =
            if h0 >= 0 then h0 + lowbit0 - lowbit1
            else if lowbit0 = 0 then h0 + lowbit1
            else h0 + lowbit0 - lowbit1
          in
          put o (i + 1) (j + 1) ((h0 + hx + hy + hc) asr shift);
          put o (i + 1) j ((h0 + hx - hy - hc) asr shift);
          put o i (j + 1) ((h0 - hx + hy - hc) asr shift);
          put o i j ((h0 - hx - hy + hc) asr shift)
        end
        else if has_x || has_y then begin
          (* a block cut by the last column keeps [hx], by the last row
             [hy] *)
          let h = round (if has_x then hx else hy) prnd1 nrnd1 mask1 in
          let lowbit1 = h land bit1 in
          let h0 = if h0 >= 0 then h0 - lowbit1 else h0 + lowbit1 in
          if has_x then put o (i + 1) j ((h0 + h) asr shift)
          else put o i (j + 1) ((h0 + h) asr shift);
          put o i j ((h0 - h) asr shift)
        end
        else put o i j (h0 asr shift)
      done
    done
  done

(* Decoding *)

let decode ~smooth:sm (src : Checksum.bigbytes) ~x0 ~x1 ~y0 ~y1 (out : ints)
    =
  let r = bits src in
  let h = header r in
  let nx = h.nx and ny = h.ny in
  let a = { keys = [||]; ors = [||]; count = 0 } in
  let nx2 = half_up nx and ny2 = half_up ny in
  quadrant r a 0 ny nx2 ny2 h.planes.(0);
  quadrant r a ny2 ny nx2 (ny / 2) h.planes.(1);
  quadrant r a (ny * nx2) ny (nx / 2) ny2 h.planes.(1);
  quadrant r a ((ny * nx2) + ny2) ny (nx / 2) (ny / 2) h.planes.(2);
  if nbits r 4 <> 0 then fail "the bit planes do not end with a zero nybble";
  (* The signs follow from the next byte, one per nonzero coefficient in
     raster order. *)
  r.left <- 0;
  let c = coefs a in
  for i = 0 to Array.length c.mag - 1 do
    if bit r = 1 then c.mag.(i) <- -c.mag.(i)
  done;
  let mult = if h.scale > 1 then h.scale else 1 in
  let coef k = if k = 0 then h.h0 * mult else find c k * mult in
  let w = win x0 x1 y0 y1 in
  invert coef nx ny ~smooth:sm h.scale w;
  Array.iteri (fun k v -> A.unsafe_set out k v) w.v;
  h.scale
