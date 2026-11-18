(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  mutable span : float array; (* Edge k's span in [lo, hi], at 2k, 2k + 1. *)
  mutable order : int array; (* The edges with a span, by its start. *)
  mutable cuts : float array;
  mutable active : int array;
  mutable keys : float array;
  mutable start : float array; (* Of edge k's open window. *)
  mutable sign : float array; (* Of edge k's open window, 0 for none. *)
  mutable count : int; (* The windows found, ... *)
  mutable window : int array; (* ... each window's edge, *)
  mutable ends : float array; (* its ends and heights there, 4i to 4i + 3, *)
  mutable signs : float array; (* and its sign. *)
}

let create () =
  {
    span = [||];
    order = [||];
    cuts = [||];
    active = [||];
    keys = [||];
    start = [||];
    sign = [||];
    count = 0;
    window = [||];
    ends = [||];
    signs = [||];
  }

let grow a n = if Array.length a >= n then a else Array.make (2 * n) 0.
let grow_int a n = if Array.length a >= n then a else Array.make (2 * n) 0

let[@inline] height e k x =
  let xp = e.(4 * k) and yp = e.((4 * k) + 1) in
  let xq = e.((4 * k) + 2) and yq = e.((4 * k) + 3) in
  if x = xp then yp
  else if x = xq then yq
  else
    let t = (x -. xp) /. (xq -. xp) in
    let t = if t < 0. then 0. else if t > 1. then 1. else t in
    yp +. (t *. (yq -. yp))

(* The first [n] elements of [a] sorted: by insertion where they are few, as in
   a column, by the library's sort otherwise. *)
let sort_floats a n =
  if n <= 32 then
    for i = 1 to n - 1 do
      let x = a.(i) in
      let j = ref (i - 1) in
      while !j >= 0 && a.(!j) > x do
        a.(!j + 1) <- a.(!j);
        decr j
      done;
      a.(!j + 1) <- x
    done
  else begin
    let b = Array.sub a 0 n in
    Array.sort Float.compare b;
    Array.blit b 0 a 0 n
  end

(* The first [n] edges of [a] sorted by the start of their span, keeping
   ties in order. *)
let sort_by_start sp a n =
  if n <= 32 then
    for i = 1 to n - 1 do
      let k = a.(i) in
      let x = sp.(2 * k) in
      let j = ref (i - 1) in
      while !j >= 0 && sp.(2 * a.(!j)) > x do
        a.(!j + 1) <- a.(!j);
        decr j
      done;
      a.(!j + 1) <- k
    done
  else begin
    let b = Array.sub a 0 n in
    Array.stable_sort (fun k l -> Float.compare sp.(2 * k) sp.(2 * l)) b;
    Array.blit b 0 a 0 n
  end

let grow_cuts s n =
  let c = Array.make (2 * (n + 1)) 0. in
  Array.blit s.cuts 0 c 0 n;
  s.cuts <- c

let[@inline] add_cut s n x =
  if n = Array.length s.cuts then grow_cuts s n;
  s.cuts.(n) <- x;
  n + 1

(* The cuts: [lo], [hi], each edge's span's ends, and each crossing of two
   edges inside their common span. The crossings come from a sweep in order of
   the spans' starts, each edge against the earlier ones whose span reaches
   past its start. Returns the number of distinct cuts, sorted. *)
let cut s e m ~lo ~hi =
  let sp = s.span and order = s.order in
  let nc = ref 0 in
  nc := add_cut s !nc lo;
  nc := add_cut s !nc hi;
  for r = 0 to m - 1 do
    let k = order.(r) in
    nc := add_cut s !nc sp.(2 * k);
    nc := add_cut s !nc sp.((2 * k) + 1)
  done;
  s.active <- grow_int s.active m;
  let active = s.active and na = ref 0 in
  for r = 0 to m - 1 do
    let k = order.(r) in
    let a = sp.(2 * k) and b = sp.((2 * k) + 1) in
    let kept = ref 0 in
    for i = 0 to !na - 1 do
      let l = active.(i) in
      let bl = sp.((2 * l) + 1) in
      if bl > a then begin
        active.(!kept) <- l;
        incr kept;
        let x0 = a and x1 = if bl < b then bl else b in
        let f0 = height e k x0 -. height e l x0
        and f1 = height e k x1 -. height e l x1 in
        if (f0 < 0. && f1 > 0.) || (f0 > 0. && f1 < 0.) then begin
          let x = x0 +. ((x1 -. x0) *. (f0 /. (f0 -. f1))) in
          if x0 < x && x < x1 then nc := add_cut s !nc x
        end
      end
    done;
    na := !kept;
    active.(!na) <- k;
    incr na
  done;
  let cuts = s.cuts in
  sort_floats cuts !nc;
  let n = ref 1 in
  for i = 1 to !nc - 1 do
    if cuts.(i) > cuts.(!n - 1) then begin
      cuts.(!n) <- cuts.(i);
      incr n
    end
  done;
  !n

let grow_windows s n =
  let more = n + 64 in
  s.window <- Array.append s.window (Array.make more 0);
  s.ends <- Array.append s.ends (Array.make (4 * more) 0.);
  s.signs <- Array.append s.signs (Array.make more 0.)

let[@inline] add_window s e k a b sign =
  let n = s.count in
  if n = Array.length s.window then grow_windows s n;
  s.window.(n) <- k;
  s.ends.(4 * n) <- a;
  s.ends.((4 * n) + 1) <- b;
  s.ends.((4 * n) + 2) <- height e k a;
  s.ends.((4 * n) + 3) <- height e k b;
  s.signs.(n) <- sign;
  s.count <- n + 1

let windows s e n ~lo ~hi =
  s.span <- grow s.span (2 * n);
  s.order <- grow_int s.order n;
  s.start <- grow s.start n;
  s.sign <- grow s.sign n;
  let sp = s.span in
  let m = ref 0 in
  for k = 0 to n - 1 do
    let xp = e.(4 * k) and xq = e.((4 * k) + 2) in
    let a = if xp < xq then xp else xq and b = if xp < xq then xq else xp in
    let a = if a < lo then lo else a and b = if b > hi then hi else b in
    sp.(2 * k) <- a;
    sp.((2 * k) + 1) <- b;
    s.sign.(k) <- 0.;
    if a < b then begin
      s.order.(!m) <- k;
      incr m
    end
  done;
  let m = !m in
  let order = s.order in
  sort_by_start sp order m;
  let ncuts = cut s e m ~lo ~hi in
  let cuts = s.cuts in
  s.active <- grow_int s.active m;
  s.keys <- grow s.keys m;
  let active = s.active and keys = s.keys in
  let na = ref 0 and next = ref 0 in
  s.count <- 0;
  for c = 0 to ncuts - 2 do
    let xa = cuts.(c) and xb = cuts.(c + 1) in
    (* The edges ending at [xa] leave, with their windows; those starting
       there join. *)
    let kept = ref 0 in
    for i = 0 to !na - 1 do
      let k = active.(i) in
      let b = sp.((2 * k) + 1) in
      if b <= xa then add_window s e k s.start.(k) b s.sign.(k)
      else begin
        active.(!kept) <- k;
        incr kept
      end
    done;
    na := !kept;
    while !next < m && sp.(2 * order.(!next)) <= xa do
      active.(!na) <- order.(!next);
      incr na;
      incr next
    done;
    (* Sorted by key, by insertion from the last slab's order. *)
    for i = 0 to !na - 1 do
      let k = active.(i) in
      let key = height e k xa +. height e k xb in
      let j = ref (i - 1) in
      while !j >= 0 && keys.(!j) > key do
        keys.(!j + 1) <- keys.(!j);
        active.(!j + 1) <- active.(!j);
        decr j
      done;
      keys.(!j + 1) <- key;
      active.(!j + 1) <- k
    done;
    for i = 0 to !na - 1 do
      let k = active.(i) in
      let sign = if i land 1 = 1 then 1. else -1. in
      if s.sign.(k) <> sign then begin
        if s.sign.(k) <> 0. then add_window s e k s.start.(k) xa s.sign.(k);
        s.start.(k) <- xa;
        s.sign.(k) <- sign
      end
    done
  done;
  for i = 0 to !na - 1 do
    let k = active.(i) in
    add_window s e k s.start.(k) sp.((2 * k) + 1) s.sign.(k)
  done;
  s.count

let edges s = s.window
let ends s = s.ends
let signs s = s.signs
