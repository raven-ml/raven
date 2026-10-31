(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A reference H-compress decoder: every coefficient and every quadtree
   entry dense, the whole tile inverted, as cfitsio's fits_hdecompress64
   runs. Costs memory and work in proportion to the tile; the tests run it
   on small tiles to check the library's decoder, which costs what the
   stream and the window hold. *)

let fail fmt = Printf.ksprintf failwith fmt

let signed32 v =
  let v = v land 0xFFFFFFFF in
  if v land 0x80000000 <> 0 then v - (1 lsl 32) else v

(* Bit input, most significant bit first *)

type bits = {
  src : string;
  mutable pos : int;
  mutable byte : int;
  mutable left : int; (* bits of [byte] not yet read *)
}

let byte r =
  if r.pos >= String.length r.src then fail "the stream ends early";
  let b = Char.code r.src.[r.pos] in
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

(* Bit planes *)

(* The least [k] with [2^k >= n]. *)
let log2_up n =
  let k = ref 0 in
  while 1 lsl !k < n do
    incr k
  done;
  !k

let half_up n = (n + 1) / 2

(* [insert s w a off n nqx nqy plane] ors [plane] into the pixels of the
   [nqx × nqy] quadrant of [a] at [off], rows [n] apart, that the 4-bit codes
   of its 2×2 blocks mark, block [(i, j)]'s code at [s.[i * w + j]]: bit 3
   for its pixel [(2i, 2j)], 2 for [(2i, 2j + 1)], 1 for [(2i + 1, 2j)] and
   0 for [(2i + 1, 2j + 1)]. *)
let insert s w (a : int array) off n nqx nqy plane =
  for x = 0 to nqx - 1 do
    for y = 0 to nqy - 1 do
      let code = Char.code (Bytes.unsafe_get s ((x / 2 * w) + (y / 2))) in
      let b = 3 - (2 * (x land 1)) - (y land 1) in
      if (code lsr b) land 1 = 1 then begin
        let k = off + (x * n) + y in
        Array.set a k (Array.get a k lor plane)
      end
    done
  done

(* [expand r s nx ny] turns the codes in [s] of the quadtree level of
   [half_up nx × half_up ny] blocks into the level of [nx × ny] below: each
   code's bits flag the four entries of its block, placed as {!insert} places
   them, and each flagged entry then reads its code, the last entry first. *)
let expand r s nx ny =
  let px = half_up nx and py = half_up ny in
  (* The levels share [s]: spread the codes to their blocks' first entries,
     the last first, since each moves to a place at or past its own. *)
  for i = px - 1 downto 0 do
    for j = py - 1 downto 0 do
      Bytes.unsafe_set s ((2 * i * ny) + (2 * j))
        (Bytes.unsafe_get s ((i * py) + j))
    done
  done;
  for i = 0 to px - 1 do
    for j = 0 to py - 1 do
      let code = Char.code (Bytes.unsafe_get s ((2 * i * ny) + (2 * j))) in
      for di = 0 to 1 do
        for dj = 0 to 1 do
          let x = (2 * i) + di and y = (2 * j) + dj in
          if x < nx && y < ny then
            Bytes.unsafe_set s
              ((x * ny) + y)
              (Char.unsafe_chr ((code lsr (3 - (2 * di) - dj)) land 1))
        done
      done
    done
  done;
  for k = (nx * ny) - 1 downto 0 do
    if Bytes.unsafe_get s k <> '\000' then
      Bytes.unsafe_set s k (Char.unsafe_chr (huffman r))
  done

(* [level levels k q] is the size along an axis of [q] entries of level [k]
   of a quadtree of [levels]: [q] halved, rounded up, [levels - k] times, at
   least 1. *)
let level levels k q =
  let d = 1 lsl (levels - k) in
  Int.max 1 ((q + d - 1) / d)

(* [quadrant r s a off n nqx nqy planes] reads the [planes] bit planes of
   the [nqx × nqy] quadrant of [a] at [off], rows [n] apart, the highest
   first. [s] holds the quadtree's widest level. A quadtree descends from
   one code over [log2_up (max nqx nqy)] levels to the quadrant's blocks. *)
let quadrant r s a off n nqx nqy planes =
  let levels = log2_up (Int.max nqx nqy) in
  let w = half_up nqy in
  for bit = planes - 1 downto 0 do
    let plane = 1 lsl bit in
    match nbits r 4 with
    | 0 ->
        for k = 0 to (half_up nqx * w) - 1 do
          Bytes.unsafe_set s k (Char.unsafe_chr (nbits r 4))
        done;
        insert s w a off n nqx nqy plane
    | 0xF ->
        Bytes.unsafe_set s 0 (Char.unsafe_chr (huffman r));
        for k = 1 to levels - 1 do
          expand r s (level levels k nqx) (level levels k nqy)
        done;
        insert s w a off n nqx nqy plane
    | c -> fail "a bit plane's code is %d, not 0 or 15" c
  done

(* Inverse H-transform *)

(* [unshuffle a tmp off n stride] interleaves the [n] values of [a] from
   [off], [stride] apart: the first [half_up n] go to the even places, the
   rest to the odd ones. *)
let unshuffle (a : int array) tmp off n stride =
  for i = 0 to n - 1 do
    Array.unsafe_set tmp i (Array.get a (off + (i * stride)))
  done;
  let half = half_up n in
  for i = 0 to n - 1 do
    let j = if i < half then 2 * i else (2 * (i - half)) + 1 in
    Array.set a (off + (j * stride)) (Array.unsafe_get tmp i)
  done

(* [smooth a nxtop nytop ny scale] moves the slopes [hx], [hy] and the
   curvature [hc] of the level's 2×2 blocks toward the values that
   interpolate their neighbours' means, within monotonicity, by at most
   half the scale, the change rounded toward zero. Blocks on the level's
   edge keep theirs. *)
let smooth (a : int array) nxtop nytop ny scale =
  let smax = scale asr 1 in
  if smax > 0 then begin
    let ny2 = 2 * ny in
    let get k = Array.get a k in
    (* [nudge k target shift] moves [a.{k}] toward [target / 2^shift]. *)
    let nudge k target shift =
      let s = target - (get k lsl shift) in
      let s =
        if s >= 0 then s asr shift else (s + (1 lsl shift) - 1) asr shift
      in
      Array.set a k (get k + Int.max (Int.min s smax) (-smax))
    in
    (* One slope from the means [hm], [h0], [hp] along an axis. *)
    let slope k hm h0 hp =
      let dmax = Int.max (Int.min (hp - h0) (h0 - hm)) 0 lsl 2 in
      let dmin = Int.min (Int.max (hp - h0) (h0 - hm)) 0 lsl 2 in
      if dmin < dmax then nudge k (Int.max (Int.min (hp - hm) dmax) dmin) 3
    in
    let i = ref 2 in
    while !i < nxtop - 2 do
      let j = ref 0 in
      while !j < nytop do
        let s00 = (ny * !i) + !j in
        slope (s00 + ny) (get (s00 - ny2)) (get s00) (get (s00 + ny2));
        j := !j + 2
      done;
      i := !i + 2
    done;
    let i = ref 0 in
    while !i < nxtop do
      let j = ref 2 in
      while !j < nytop - 2 do
        let s00 = (ny * !i) + !j in
        slope (s00 + 1) (get (s00 - 2)) (get s00) (get (s00 + 2));
        j := !j + 2
      done;
      i := !i + 2
    done;
    let i = ref 2 in
    while !i < nxtop - 2 do
      let j = ref 2 in
      while !j < nytop - 2 do
        let s00 = (ny * !i) + !j in
        let s10 = s00 + ny in
        let hmm = get (s00 - ny2 - 2) and hpm = get (s00 + ny2 - 2) in
        let hmp = get (s00 - ny2 + 2) and hpp = get (s00 + ny2 + 2) in
        let h0 = get s00 in
        let hx2 = get s10 lsl 1 and hy2 = get (s00 + 1) lsl 1 in
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
          nudge (s10 + 1)
            (Int.max (Int.min (hpp + hmm - hmp - hpm) dmax) dmin)
            6;
        j := !j + 2
      done;
      i := !i + 2
    done
  end

(* [round v pos neg mask] rounds [v] to the multiple that [mask] keeps,
   adding [pos] to a nonnegative [v] and [neg] to a negative one. *)
let round v pos neg mask = (v + if v >= 0 then pos else neg) land mask

(* [inverse a nx ny ~smooth:sm scale] inverts the H-transform of the
   [nx × ny] coefficients in [a], level by level from the coarsest: a
   level's coefficients interleave, each 2×2 block's [h0], [hx], [hy], [hc]
   round to the multiples their level's divisions left, the low bits the
   forward transform discarded pass from [hc], [hx] and [hy] to [h0], and
   the block's four sums halve (quarter on the last level). A tile of one
   pixel has no level. *)
let inverse (a : int array) nx ny ~smooth:sm scale =
  let nmax = Int.max nx ny in
  let levels = log2_up nmax in
  if levels > 0 then begin
    let tmp = Array.make nmax 0 in
    let bit0 = ref (1 lsl (levels - 1)) in
    let bit1 = ref (!bit0 lsl 1) in
    let mask0 = ref (- !bit0) in
    let mask1 = ref (!mask0 lsl 1) in
    let prnd0 = ref (!bit0 asr 1) and prnd1 = ref (!bit1 asr 1) in
    let nrnd0 = ref (!prnd0 - 1) and nrnd1 = ref (!prnd1 - 1) in
    let bit2 = !bit0 lsl 2 in
    let prnd2 = bit2 asr 1 in
    Array.set a 0 (round (Array.get a 0) prnd2 (prnd2 - 1) (- bit2));
    let shift = ref 1 in
    for k = levels - 1 downto 0 do
      let d = 1 lsl k in
      let nxtop = (nx + d - 1) / d and nytop = (ny + d - 1) / d in
      if k = 0 then begin
        nrnd0 := 0;
        shift := 2
      end;
      for i = 0 to nxtop - 1 do
        unshuffle a tmp (ny * i) nytop 1
      done;
      for j = 0 to nytop - 1 do
        unshuffle a tmp j nxtop ny
      done;
      if sm then smooth a nxtop nytop ny scale;
      let oddx = nxtop land 1 and oddy = nytop land 1 in
      let sh = !shift in
      let get k = Array.get a k and set k v = Array.set a k v in
      (* [h0] with the low bit [lowbit1] of an edge's one difference *)
      let edge s00 s1 =
        let h0 = get s00 in
        let h = round (get s1) !prnd1 !nrnd1 !mask1 in
        let lowbit1 = h land !bit1 in
        let h0 = if h0 >= 0 then h0 - lowbit1 else h0 + lowbit1 in
        set s1 ((h0 + h) asr sh);
        set s00 ((h0 - h) asr sh)
      in
      let i = ref 0 in
      while !i < nxtop - oddx do
        let j = ref 0 in
        while !j < nytop - oddy do
          let s00 = (ny * !i) + !j in
          let s10 = s00 + ny in
          let h0 = get s00 in
          let hx = round (get s10) !prnd1 !nrnd1 !mask1 in
          let hy = round (get (s00 + 1)) !prnd1 !nrnd1 !mask1 in
          let hc = round (get (s10 + 1)) !prnd0 !nrnd0 !mask0 in
          let lowbit0 = hc land !bit0 in
          let hx = if hx >= 0 then hx - lowbit0 else hx + lowbit0 in
          let hy = if hy >= 0 then hy - lowbit0 else hy + lowbit0 in
          let lowbit1 = (hc lxor hx lxor hy) land !bit1 in
          let h0 =
            if h0 >= 0 then h0 + lowbit0 - lowbit1
            else if lowbit0 = 0 then h0 + lowbit1
            else h0 + lowbit0 - lowbit1
          in
          set (s10 + 1) ((h0 + hx + hy + hc) asr sh);
          set s10 ((h0 + hx - hy - hc) asr sh);
          set (s00 + 1) ((h0 - hx + hy - hc) asr sh);
          set s00 ((h0 - hx - hy + hc) asr sh);
          j := !j + 2
        done;
        (* the last entry of an odd row, alone with [hx] *)
        if oddy = 1 then begin
          let s00 = (ny * !i) + !j in
          edge s00 (s00 + ny)
        end;
        i := !i + 2
      done;
      (* the last row of an odd column, alone with [hy] *)
      if oddx = 1 then begin
        let j = ref 0 in
        while !j < nytop - oddy do
          let s00 = (ny * !i) + !j in
          edge s00 (s00 + 1);
          j := !j + 2
        done;
        if oddy = 1 then begin
          let s00 = (ny * !i) + !j in
          set s00 (get s00 asr sh)
        end
      end;
      bit1 := !bit0;
      bit0 := !bit0 asr 1;
      mask1 := !mask0;
      mask0 := !mask0 asr 1;
      prnd1 := !prnd0;
      prnd0 := !prnd0 asr 1;
      nrnd1 := !nrnd0;
      nrnd0 := !prnd0 - 1
    done
  end

(* Decoding *)

let decode ~smooth:sm src n =
  let r = { src; pos = 0; byte = 0; left = 0 } in
  let m0 = byte r in
  let m1 = byte r in
  if (m0, m1) <> (0xDD, 0x99) then fail "the stream does not start with DD 99";
  let nx = signed32 (be r 4) in
  let ny = signed32 (be r 4) in
  let scale = signed32 (be r 4) in
  if nx < 1 || ny < 1 || nx * ny <> n then
    fail "the stream holds %d x %d pixels, the tile %d" nx ny n;
  let h = ref 0L in
  for _ = 1 to 8 do
    h := Int64.logor (Int64.shift_left !h 8) (Int64.of_int (byte r))
  done;
  let planes = Array.init 3 (fun _ -> byte r) in
  Array.iter (fun p -> if p > 62 then fail "%d bit planes" p) planes;
  let out = Array.make n 0 in
  let nx2 = half_up nx and ny2 = half_up ny in
  let s =
    Bytes.make (Int.max 1 (half_up nx2) * Int.max 1 (half_up ny2)) '\000'
  in
  quadrant r s out 0 ny nx2 ny2 planes.(0);
  quadrant r s out ny2 ny nx2 (ny / 2) planes.(1);
  quadrant r s out (ny * nx2) ny (nx / 2) ny2 planes.(1);
  quadrant r s out ((ny * nx2) + ny2) ny (nx / 2) (ny / 2) planes.(2);
  if nbits r 4 <> 0 then fail "the bit planes do not end with a zero nybble";
  r.left <- 0;
  for i = 0 to n - 1 do
    let v = out.(i) in
    if v <> 0 && bit r = 1 then out.(i) <- -v
  done;
  out.(0) <- Int64.to_int !h;
  if scale > 1 then
    for i = 0 to n - 1 do
      out.(i) <- out.(i) * scale
    done;
  inverse out nx ny ~smooth:sm scale;
  (nx, ny, out, scale)
