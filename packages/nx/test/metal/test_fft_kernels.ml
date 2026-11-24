(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's Fourier transforms' kernels on the machine's Metal GPU 0, skipping
   without one: each work a call plans, run alone on drawn operands, writes what
   fft.mli says it writes, bit for bit, against a host reading of the launch's
   semantics with float32 arithmetic emulated exactly. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module F = Nx_metal.Fft
module T = Nx_metal.Fft_tables
module P = Nx_metal.Plan
module S = Nx_metal_support

let answer =
  Testable.make ~pp:Nx_array_support.pp_answer ~equal:(fun a b -> a = b)

(* The twiddle's low table: [2^low_bits] roots. *)
let low_bits = Nx_metal.Kernels.fft_low_bits
let low_mask = (1 lsl low_bits) - 1

(* Float32 arithmetic *)

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let least_normal = Float.ldexp 1. (-126)

(* Apple7 takes a subnormal operand as a zero of its sign, and gives a zero of
   its sign for a result whose exact value lies below the least normal float32,
   before rounding. *)
let flush x = if Float.abs x < least_normal then Float.copy_sign 0. x else x

(* [x + c] rounded once to float32, [x] and [c] exact in a double: their sum as
   a double and its error, rounded to the nearest float32, the error deciding a
   sum halfway between two. *)
let two_sum x c =
  let s = x +. c in
  let bb = s -. x in
  (s, x -. (s -. bb) +. (c -. bb))

(* Whether [s + e], [e] within half of [s]'s last place, lies below the least
   normal float32. *)
let tiny s e =
  let m = Float.abs s in
  m < least_normal || (m = least_normal && e <> 0. && e > 0. <> (s > 0.))

let rounded x c =
  let s, e = two_sum x c in
  let r = f32 s in
  let d = s -. r in
  if e = 0. || d = 0. then r
  else
    let other = r +. (2. *. d) in
    if f32 other = other && e > 0. = (d > 0.) then other else r

(* [x + c] as Apple7 gives it, [x] and [c] exact in a double, as the product of
   two float32s is. *)
let summed x c =
  let s, e = two_sum x c in
  if tiny s e then Float.copy_sign 0. s else rounded x c

let mul a b =
  let p = flush a *. flush b in
  if Float.abs p < least_normal then Float.copy_sign 0. p else f32 p

let add a b = summed (flush a) (flush b)
let sub a b = add a (-.b)
let fma a b c = summed (flush a *. flush b) (flush c)

(* fft.metal's complex product: two products and two fused multiply-adds. *)
let cmul (ar, ai) (wr, wi) = (fma ar wr (-.mul ai wi), fma ar wi (mul ai wr))
let conj (r, i) = (r, -.i)

(* Bytes *)

let get b at = Int32.float_of_bits (Bytes.get_int32_le b at)
let put b at x = Bytes.set_int32_le b at (Int32.bits_of_float x)

let entry (t : T.bigbytes) i =
  let f at =
    let v = ref 0l in
    for k = 3 downto 0 do
      v :=
        Int32.logor (Int32.shift_left !v 8)
          (Int32.of_int (Bigarray.Array1.get t (at + k)))
    done;
    Int32.float_of_bits !v
  in
  (f (8 * i), f ((8 * i) + 4))

(* Entry [i] of a table of int32s. *)
let int_at (t : T.bigbytes) i =
  let v = ref 0 in
  for k = 3 downto 0 do
    v := (!v lsl 8) lor Bigarray.Array1.get t ((4 * i) + k)
  done;
  !v

(* Tables

   The model asks for a line's tables at each launch it checks, in both
   directions; a double-double table costs far more than the launch, so each is
   made once a process. *)

let once f =
  let made = Hashtbl.create 16 in
  fun k ->
    match Hashtbl.find_opt made k with
    | Some t -> t
    | None ->
        let t = f k in
        Hashtbl.add made k t;
        t

module Tables = struct
  let global = once T.global
  let chirp = once T.chirp
  let rader = once T.rader
  let bluestein = once (fun (n, m) -> T.bluestein ~n ~m)
  let roots = once (fun (l, count) -> T.roots l ~start:0 ~step:1 ~count)
end

(* Sides *)

(* The position of line [l]'s first point: its coordinates over the side's lines
   in C order, each times its stride. *)
let line_at (s : F.side) l =
  let at = ref s.first and l = ref l in
  for k = Array.length s.dims - 1 downto 0 do
    at := !at + (!l mod s.dims.(k) * s.strides.(k));
    l := !l / s.dims.(k)
  done;
  !at

let lines (s : F.side) = Array.fold_left ( * ) 1 s.dims

(* The bytes of point [j]'s parts of line [l], from its buffer's first byte: a
   complex element, or a pair of floats of a 4-byte side. *)
let origin (s : F.side) =
  match s.buffer with Workspace at -> at | Operand | Dst -> 0

let parts (s : F.side) l j =
  let at = line_at s l and o = origin s in
  if s.width = 8 then
    let p = o + (8 * (at + (j * s.ps))) in
    (p, p + 4)
  else (o + (4 * (at + (2 * j * s.ps))), o + (4 * (at + (((2 * j) + 1) * s.ps))))

let float_at (s : F.side) l j = origin s + (4 * (line_at s l + (j * s.ps)))

(* The axes a side's words hold. *)
let held = Nx_metal.Kernels.fft_side_holds

(* Whether a transpose copies bits, [Complex] or [Real] to [Complex] or [Real]
   but [Real] to [Complex], which an inverse runs as forward. *)
let copies (t : F.work) =
  match t with
  | Transpose { scale = Plain; input = Complex | Real; output = Real; _ }
  | Transpose { scale = Plain; input = Complex; output = Complex; _ } ->
      true
  | Transpose _ | Split _ | Prepass _ | Step _ | Column_step _ -> false

(* An inverse launch (fft.mli) is [conj ∘ forward ∘ conj] on its sides: its
   forward operation on its source's points conjugated, its written points
   conjugated, but in a copy. The models inline the two flips at each load and
   store, as the kernels do. *)
let flips ~inverse (l : F.launch) = inverse && not (copies l.work)

(* A transpose's semantics (fft.mli): over [a] rows of [b] each run, a line of
   both sides, takes point [j = r b + c] to point [c a + r]; a move takes each
   line's point [j] to the same; between, its scale, and what it reads and
   writes, the line's [n] points bounding the chirp, the bins and the halves; in
   an inverse, each loaded and each stored point conjugated, a read's [+0]
   imaginary parts set after the load's. *)
let transposed ~inverse (l : F.launch) ~src ~dst =
  let flip z = if flips ~inverse l then conj z else z in
  match l.work with
  | Transpose { split = a, b; input; scale; output; line; _ } ->
      let n = line.n in
      let table =
        match scale with
        | Plain -> None
        | Twiddle ->
            let lo, hi = Tables.global (a * b) in
            Some (lo, hi)
        | Chirp_in | Chirp_out ->
            let c = Tables.chirp n in
            Some (c, c)
        | Kernel -> (
            match line.how with
            | Bluestein m ->
                let k = Tables.bluestein (n, m) in
                Some (k, k)
            | Smooth | Rader -> failwith "a kernel of a line not Bluestein's")
      in
      let lo i = entry (fst (Option.get table)) i
      and hi i = entry (snd (Option.get table)) i in
      let read line j =
        if scale = Chirp_in && j >= n then (0., 0.)
        else
          let z =
            match input with
            | Complex ->
                let r, i = parts l.src line j in
                flip (get src r, get src i)
            | Real -> (get src (float_at l.src line j), 0.)
            | Bins ->
                let at k =
                  let r, i = parts l.src line k in
                  flip (get src r, get src i)
                in
                if j = 0 then (fst (at 0), 0.)
                else if j <= n / 2 then at j
                else conj (at (n - j))
          in
          if scale = Chirp_in then cmul z (lo j) else z
      in
      let write line k ~r ~c z =
        let kept =
          (scale <> Chirp_out || k < n) && (output <> Half || k <= n / 2)
        in
        let z =
          if not kept then z
          else
            match scale with
            | Twiddle ->
                let e = r * c in
                cmul z (cmul (lo (e land low_mask)) (hi (e lsr low_bits)))
            | Kernel -> conj (cmul z (lo k))
            | Chirp_out -> cmul (conj z) (lo k)
            | Plain | Chirp_in -> z
        in
        let z = flip z in
        if kept then
          match output with
          | Complex | Half ->
              let pr, pi = parts l.dst line k in
              put dst pr (fst z);
              put dst pi (snd z)
          | Real -> put dst (float_at l.dst line k) (fst z)
      in
      if a = 1 then
        for line = 0 to lines l.src - 1 do
          for j = 0 to b - 1 do
            write line j ~r:line ~c:j (read line j)
          done
        done
      else
        for line = 0 to lines l.src - 1 do
          for r = 0 to a - 1 do
            for c = 0 to b - 1 do
              write line ((c * a) + r) ~r ~c (read line ((r * b) + c))
            done
          done
        done
  | Step _ | Column_step _ | Split _ | Prepass _ -> failwith "not a transpose"

(* A split's or a pre-pass's semantics (kernels.h): each line's pair [k <= h /
   2] with the forward root [w = w_2h^k], from the twiddle's two tables. A
   split, in place, from [a = Z_k] and [b = Z_(h-k)] ([Z_0] at [k = 0]) writes
   [X_k] then [X_(h-k)]; a pre-pass from bins [a = X_k] and [b = X_(h-k)], their
   imaginary parts [+0] at [k = 0], writes [Z'_k], and [Z'_(h-k)] where [0 < k]
   and [2 k <> h]. In an inverse each loaded and each stored point is
   conjugated, the [+0] set after the load's. *)
let paired ~inverse (l : F.launch) ~src ~dst =
  let split = match l.work with Split _ -> true | _ -> false in
  let bins = if split then l.dst else l.src in
  let h = bins.points - 1 in
  let lo, hi = Tables.global (2 * h) in
  let half = 0.5 in
  let flip z = if inverse then conj z else z in
  let load b s line k =
    let r, i = parts s line k in
    flip (get b r, get b i)
  and store s line k z =
    let zr, zi = flip z in
    let r, i = parts s line k in
    put dst r zr;
    put dst i zi
  in
  for line = 0 to lines bins - 1 do
    for k = 0 to h / 2 do
      let w = cmul (entry lo (k land low_mask)) (entry hi (k lsr low_bits)) in
      let from = if split then dst else src
      and side = if split then l.dst else l.src in
      let a = load from side line k
      and b = load from side line (if split && k = 0 then 0 else h - k) in
      let a, b =
        if (not split) && k = 0 then ((fst a, 0.), (fst b, 0.)) else (a, b)
      in
      let bc = conj b in
      let er = add (fst a) (fst bc) and ei = add (snd a) (snd bc) in
      let dr = sub (fst a) (fst bc) and di = sub (snd a) (snd bc) in
      let wr, wi = w in
      if split then begin
        let u = fma wr di (mul wi dr) and v = fma (-.wr) dr (mul wi di) in
        store l.dst line k (mul half (add er u), mul half (add ei v));
        store l.dst line (h - k) (mul half (sub er u), mul half (sub v ei))
      end
      else begin
        let u = fma wr di (mul wi dr) and v = fma (-.wr) dr (mul wi di) in
        store l.dst line k (add er u, add ei v);
        if k <> 0 && 2 * k <> h then
          store l.dst line (h - k) (sub er u, sub v ei)
      end
    done
  done

(* Steps *)

let cadd (ar, ai) (br, bi) = (add ar br, add ai bi)
let csub (ar, ai) (br, bi) = (sub ar br, sub ai bi)

(* A DFT of 4 (kernels.h, steps). *)
let dft4 a0 a1 a2 a3 =
  let t0 = cadd a0 a2 and t1 = csub a0 a2 in
  let t2 = cadd a1 a3 and t3 = csub a1 a3 in
  ( cadd t0 t2,
    (add (fst t1) (snd t3), sub (snd t1) (fst t3)),
    csub t0 t2,
    (sub (fst t1) (snd t3), add (snd t1) (fst t3)) )

let c8 = f32 0.70710678118654752440
let c16 = f32 0.92387953251128675613
let s16 = f32 0.38268343236508977173

(* [x w] for [w = (cc, -ss)]; [x w_16^2] and [x w_16^6]. *)
let w16 (xr, xi) cc ss = (fma xr cc (mul xi ss), fma xi cc (-.mul xr ss))
let w16_2 (xr, xi) = (mul c8 (add xr xi), mul c8 (sub xi xr))
let w16_6 (xr, xi) = (mul c8 (sub xi xr), -.mul c8 (add xr xi))

(* A DFT of 16 in place (kernels.h, steps): a DFT of 4 over [n2] of [v.(n1 + 4
   n2)], the twiddles [w_16^(n1 k2)], a DFT of 4 over [n1] into [v.(k2 + 4
   k1)]. *)
let dft16 v =
  for n1 = 0 to 3 do
    let a0, a1, a2, a3 = dft4 v.(n1) v.(n1 + 4) v.(n1 + 8) v.(n1 + 12) in
    v.(n1) <- a0;
    v.(n1 + 4) <- a1;
    v.(n1 + 8) <- a2;
    v.(n1 + 12) <- a3
  done;
  v.(5) <- w16 v.(5) c16 s16;
  v.(9) <- w16_2 v.(9);
  v.(13) <- w16 v.(13) s16 c16;
  v.(6) <- w16_2 v.(6);
  v.(10) <- (snd v.(10), -.fst v.(10));
  v.(14) <- w16_6 v.(14);
  v.(7) <- w16 v.(7) s16 c16;
  v.(11) <- w16_6 v.(11);
  (let r, i = w16 v.(15) c16 s16 in
   v.(15) <- (-.r, -.i));
  let u = Array.copy v in
  for k2 = 0 to 3 do
    let a0, a1, a2, a3 =
      dft4 u.(4 * k2) u.((4 * k2) + 1) u.((4 * k2) + 2) u.((4 * k2) + 3)
    in
    v.(k2) <- a0;
    v.(k2 + 4) <- a1;
    v.(k2 + 8) <- a2;
    v.(k2 + 12) <- a3
  done

let step_points = Nx_metal.Kernels.fft_step_points

(* The step of 4096 points' FFT of a line [x] (kernels.h, steps): three passes
   of 16 by 256 threads, the inner table's roots [w_4096^e], in an inverse each
   loaded and each stored point conjugated. *)
let s4096 ~inverse x =
  let flip z = if inverse then conj z else z in
  let lo = Tables.roots (step_points, step_points / 2) in
  let root e =
    if e < step_points / 2 then entry lo e
    else
      let r, i = entry lo (e - (step_points / 2)) in
      (-.r, -.i)
  in
  let threads = step_points / 16 in
  let plane = Array.make step_points (0., 0.) in
  for t = 0 to threads - 1 do
    let v = Array.init 16 (fun r -> flip x.(t + (threads * r))) in
    dft16 v;
    Array.iteri (fun r z -> plane.((16 * t) + r) <- z) v
  done;
  let pass ns write =
    let next = Array.make step_points (0., 0.) in
    for t = 0 to threads - 1 do
      let v = Array.init 16 (fun r -> plane.(t + (threads * r))) in
      let e = step_points / (16 * ns) * (t mod ns) in
      for r = 1 to 15 do
        v.(r) <- cmul v.(r) (root (e * r))
      done;
      dft16 v;
      Array.iteri (fun r z -> next.(write t r) <- z) v
    done;
    next
  in
  let p1 = pass 16 (fun t r -> (t / 16 * 256) + (t mod 16) + (16 * r)) in
  Array.blit p1 0 plane 0 step_points;
  Array.map flip (pass 256 (fun t r -> t + (threads * r)))

(* A step's semantics: each line of its source side [a], its [fft], to the same
   line of its destination side [b]. *)
let lines_through fft (a : F.side) (b : F.side) ~src ~dst =
  for line = 0 to lines a - 1 do
    let x =
      Array.init a.points (fun j ->
          let r, i = parts a line j in
          (get src r, get src i))
    in
    Array.iteri
      (fun k (zr, zi) ->
        let r, i = parts b line k in
        put dst r zr;
        put dst i zi)
      (fft x)
  done

let stepped_on ~inverse a b ~src ~dst =
  lines_through (s4096 ~inverse) a b ~src ~dst

let stepped ~inverse (l : F.launch) ~src ~dst =
  stepped_on ~inverse l.src l.dst ~src ~dst

(* Specialised steps *)

(* A DFT of 2, of 8 and of an odd prime (kernels.h, A specialised step). *)
let dft2 v =
  let a = v.(0) and b = v.(1) in
  v.(0) <- cadd a b;
  v.(1) <- csub a b

let dft8 v =
  let b = Array.init 4 (fun k -> cadd v.(k) v.(k + 4)) in
  let d = Array.init 4 (fun k -> csub v.(k) v.(k + 4)) in
  let b0, b1, b2, b3 = dft4 b.(0) b.(1) b.(2) b.(3) in
  v.(0) <- b0;
  v.(2) <- b1;
  v.(4) <- b2;
  v.(6) <- b3;
  let (d0r, d0i), (d1r, d1i), (d2r, d2i), (d3r, d3i) =
    (d.(0), d.(1), d.(2), d.(3))
  in
  let t0r = add d0r d2i and t0i = sub d0i d2r in
  let t1r = sub d0r d2i and t1i = add d0i d2r in
  let p1 = add d1r d1i and q1 = sub d1i d1r in
  let p3 = sub d3i d3r and q3 = -.add d3r d3i in
  let sr = add p1 p3 and si = add q1 q3 in
  let dr = sub p1 p3 and di = sub q1 q3 in
  v.(1) <- (fma c8 sr t0r, fma c8 si t0i);
  v.(5) <- (fma (-.c8) sr t0r, fma (-.c8) si t0i);
  v.(3) <- (fma c8 di t1r, fma (-.c8) dr t1i);
  v.(7) <- (fma (-.c8) di t1r, fma c8 dr t1i)

(* cos (2π m / p) and sin (2π m / p), 0 < m < p, as the float32 nearest each. *)
let odd_cos p m =
  f32
    (match (p, m) with
    | 3, _ -> -0.5
    | 5, (1 | 4) -> 0.30901699437494742410
    | 5, _ -> -0.80901699437494742410
    | 7, (1 | 6) -> 0.62348980185873353053
    | 7, (2 | 5) -> -0.22252093395631440429
    | _ -> -0.90096886790241912624)

let odd_sin p m =
  f32
    (match (p, m) with
    | 3, 1 -> 0.86602540378443864676
    | 3, _ -> -0.86602540378443864676
    | 5, 1 -> 0.95105651629515357212
    | 5, 2 -> 0.58778525229247312917
    | 5, 3 -> -0.58778525229247312917
    | 5, _ -> -0.95105651629515357212
    | 7, 1 -> 0.78183148246802980871
    | 7, 2 -> 0.97492791218182360702
    | 7, 3 -> 0.43388373911755812048
    | 7, 4 -> -0.43388373911755812048
    | 7, 5 -> -0.97492791218182360702
    | _ -> -0.78183148246802980871)

let dft_odd p v =
  let h = (p - 1) / 2 in
  let t =
    Array.init (h + 1) (fun j ->
        if j = 0 then (0., 0.) else cadd v.(j) v.(p - j))
  in
  let d =
    Array.init (h + 1) (fun j ->
        if j = 0 then (0., 0.) else csub v.(j) v.(p - j))
  in
  let a0 = v.(0) in
  let y0 = ref a0 in
  for j = 1 to h do
    y0 := cadd !y0 t.(j)
  done;
  v.(0) <- !y0;
  for k = 1 to h do
    let m = ref a0 in
    for j = 1 to h do
      let cj = odd_cos p (j * k mod p) and mr, mi = !m and tr, ti = t.(j) in
      m := (fma cj tr mr, fma cj ti mi)
    done;
    let s1 = odd_sin p (k mod p) and d1r, d1i = d.(1) in
    let n = ref (mul s1 d1r, mul s1 d1i) in
    for j = 2 to h do
      let sj = odd_sin p (j * k mod p) and nr, ni = !n and dr, di = d.(j) in
      n := (fma sj dr nr, fma sj di ni)
    done;
    let (mr, mi), (nr, ni) = (!m, !n) in
    v.(k) <- (add mr ni, sub mi nr);
    v.(p - k) <- (sub mr ni, add mi nr)
  done

let dft r v =
  match r with
  | 2 -> dft2 v
  | 4 ->
      let a0, a1, a2, a3 = dft4 v.(0) v.(1) v.(2) v.(3) in
      v.(0) <- a0;
      v.(1) <- a1;
      v.(2) <- a2;
      v.(3) <- a3
  | 8 -> dft8 v
  | 16 -> dft16 v
  | p -> dft_odd p v

(* A specialised step's FFT of a line [x] (kernels.h, A specialised step): its
   passes in order, each butterfly's points twiddled past the first by the inner
   table's roots, in an inverse each loaded and each stored point conjugated; a
   step of one point, each part times the table's 1. *)
let key_fft ~inverse (step : F.step) x =
  let l = step.l in
  let count = if l mod 2 = 0 then l / 2 else l in
  let lo = Tables.roots (l, count) in
  if Array.length step.radices = 0 then
    let one = fst (entry lo 0) in
    Array.map (fun (r, i) -> (mul r one, mul i one)) x
  else
    let root e =
      if l mod 2 <> 0 || e < l / 2 then entry lo e
      else
        let r, i = entry lo (e - (l / 2)) in
        (-.r, -.i)
    in
    let flip z = if inverse then conj z else z in
    let cur = ref (Array.map flip x) and ns = ref 1 in
    Array.iteri
      (fun pass r ->
        let q = l / r and ns' = !ns in
        let next = Array.make l (0., 0.) in
        for b = 0 to q - 1 do
          let v = Array.init r (fun i -> !cur.(b + (q * i))) in
          if pass > 0 then begin
            let e = b mod ns' * (l / (ns' * r)) in
            for i = 1 to r - 1 do
              v.(i) <- cmul v.(i) (root (e * i))
            done
          end;
          dft r v;
          let at = (b / ns' * ns' * r) + (b mod ns') in
          Array.iteri (fun i z -> next.(at + (ns' * i)) <- z) v
        done;
        cur := next;
        ns := ns' * r)
      step.radices;
    Array.map flip !cur

(* A line's convolution in its group (kernels.h, A specialised step), of its
   points [x] as loaded, by the step's [fft]: Rader's, of [n = l + 1] points,
   permutes [x_(g^s)] to [s], runs [fft], takes [y_0 = A_0 + x_0] and [conj (A_k
   B_k)], runs [fft] again, and gives point [p > 0] [x_0 + conj c_q], [q = (l -
   log p) mod l]; Bluestein's, of [n] points, takes [x_j chirp_j], [+0] past
   [n], runs [fft], takes [conj (Z_k B_k)], runs [fft] again, and gives point [k
   < n] [conj (c_k) chirp_k]. The tables are made once. *)
let convolved (line : F.line) fft =
  let n = line.n in
  match line.how with
  | Smooth -> fft
  | Rader ->
      let kernel, logs = Tables.rader n in
      let l = n - 1 in
      fun x ->
        let a = Array.make l (0., 0.) in
        for p = 1 to l do
          a.(int_at logs p) <- x.(p)
        done;
        let big = fft a in
        let y0 = cadd big.(0) x.(0) in
        let c =
          fft (Array.mapi (fun k z -> conj (cmul z (entry kernel k))) big)
        in
        Array.init n (fun p ->
            if p = 0 then y0
            else
              let s = int_at logs p in
              cadd x.(0) (conj c.(if s = 0 then 0 else l - s)))
  | Bluestein m ->
      let chirp = Tables.chirp n and kernel = Tables.bluestein (n, m) in
      fun x ->
        let z =
          fft
            (Array.init m (fun j ->
                 if j < n then cmul x.(j) (entry chirp j) else (0., 0.)))
        in
        let c =
          fft (Array.mapi (fun k z -> conj (cmul z (entry kernel k))) z)
        in
        Array.init n (fun k -> cmul (conj c.(k)) (entry chirp k))

(* A specialised step's semantics by what its sides hold (kernels.h, A
   specialised step): each line's points loaded, conjugated in an inverse, its
   FFT, or its line's convolution in its group, and its points stored,
   conjugated in an inverse. A float side holds a float a point, its imaginary
   part [+0] as loaded and its real part as stored, or a pair of floats a point
   for a halved line. A c2r's odd line loads its bins [k <= n / 2], bin 0's
   imaginary part [+0], and [conj X_(n-k)] past them; an r2c's stores bins [k <=
   n / 2] alone; an r2c's line of no point loads [+0]. A halved line's lone
   step's pair [k <= h / 2], [h = n], takes the root [w = w_2h^k] from one
   table: an r2c's splits its transform's [Z_k] and [Z_(h-k)], [Z_0] at [k = 0],
   into bins [k] then [h - k]; a c2r's makes [Z'_k] and, where [0 < k] and [2 k
   <> h], [Z'_(h-k)] from bins [k] and [h - k], conjugated in an inverse, their
   imaginary parts [+0] at [k = 0], and its transform takes [Z']. *)
let keyed ~inverse (l : F.launch) (step : F.step) ~src ~dst =
  let line =
    match l.work with Step { line; _ } -> line | _ -> failwith "a step"
  in
  let real = line.real && Array.length line.steps = 1 in
  let a = l.src and b = l.dst in
  let h = if line.how = Smooth then step.l else line.n in
  let reads = a.width = 4 and writes = b.width = 4 in
  let flip z = if inverse then conj z else z in
  let roots = lazy (Tables.roots (2 * h, (h / 2) + 1)) in
  let pair k (xr, xi) (yr, yi) =
    let wr, wi = entry (Lazy.force roots) k in
    let br = yr and bi = -.yi in
    let dr = sub xr br and di = sub xi bi in
    (add xr br, add xi bi, fma wr di (mul wi dr), fma (-.wr) dr (mul wi di))
  in
  let transform = convolved line (key_fft ~inverse:false step) in
  for row = 0 to lines a - 1 do
    let complex j =
      let r, i = parts a row j in
      (get src r, get src i)
    in
    let load j =
      if a.points = 0 then (0., 0.)
      else if reads && real then complex j
      else if reads then (get src (float_at a row j), 0.)
      else if writes && (not real) && j > h / 2 then conj (complex (h - j))
      else if writes && (not real) && j = 0 then (fst (complex 0), 0.)
      else complex j
    in
    let store j (zr, zi) =
      if writes && real then begin
        let r, i = parts b row j in
        put dst r zr;
        put dst i zi
      end
      else if writes then put dst (float_at b row j) zr
      else if reads && (not real) && j > h / 2 then ()
      else begin
        let r, i = parts b row j in
        put dst r zr;
        put dst i zi
      end
    in
    if real && not reads then begin
      let made = Array.make h (0., 0.) in
      for k = 0 to h / 2 do
        let x = flip (complex k)
        and y = flip (complex (if k = 0 then h else h - k)) in
        let x, y = if k = 0 then ((fst x, 0.), (fst y, 0.)) else (x, y) in
        let er, ei, u, v = pair k x y in
        made.(k) <- (add er u, add ei v);
        if k <> 0 && 2 * k <> h then made.(h - k) <- (sub er u, sub v ei)
      done;
      Array.iteri (fun j z -> store j (flip z)) (transform made)
    end
    else begin
      let z = transform (Array.init h (fun j -> flip (load j))) in
      if real then
        for k = 0 to h / 2 do
          let er, ei, u, v = pair k z.(k) z.(if k = 0 then 0 else h - k) in
          store k (flip (mul 0.5 (add er u), mul 0.5 (add ei v)));
          store (h - k) (flip (mul 0.5 (sub er u), mul 0.5 (sub v ei)))
        done
      else Array.iteri (fun j z -> store j (flip z)) z
    end
  done

(* Calls *)

type case = {
  name : string;
  kind : [ `C2c | `R2c | `C2r ];
  axes : int array;
  x : L.t;
  dst : L.t;
}

let v ?(offset = 0) strides shape =
  L.v ~offset ~strides:(Iarray.of_array strides) (Iarray.of_array shape)

let packed shape =
  let r = Array.length shape in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  v strides shape

let c2c name ?(axes = [| 0 |]) x =
  { name; kind = `C2c; axes; x; dst = packed (Iarray.to_array (L.shape x)) }

let r2c name ?(axes = [| 0 |]) x =
  let shape = Iarray.to_array (L.shape x) in
  let last = axes.(Array.length axes - 1) in
  let d = Array.copy shape in
  d.(last) <- (shape.(last) / 2) + 1;
  { name; kind = `R2c; axes; x; dst = packed d }

let c2r name ?(axes = [| 0 |]) ~n x =
  let shape = Iarray.to_array (L.shape x) in
  let last = axes.(Array.length axes - 1) in
  let d = Array.copy shape in
  d.(last) <- n;
  { name; kind = `C2r; axes; x; dst = packed d }

let plan c =
  let f = match c.kind with `C2c -> F.c2c | `R2c -> F.r2c | `C2r -> F.c2r in
  match f ~bases:(0, 0) ~axes:c.axes c.x ~dst:c.dst with
  | Some call -> call
  | None -> failwith (c.name ^ ": declined")

(* Strides of a batch of [extents], the innermost [first], each axis one element
   past the span of those inside it, so that none merges. *)
let gapped ~first extents =
  let r = Array.length extents in
  let s = Array.make r first in
  for i = r - 2 downto 0 do
    s.(i) <- (s.(i + 1) * extents.(i + 1)) + 1
  done;
  s

(* Lines of [n] points along the last axis of a batch of [extents] that do not
   merge. *)
let along_last n extents =
  v
    (Array.append (gapped ~first:(n + 1) extents) [| 1 |])
    (Array.append extents [| n |])

(* Lines of [n] points along the first axis, scattered over a batch of [extents]
   that do not merge. *)
let along_first n extents =
  let st = gapped ~first:2 extents in
  v
    (Array.append [| (st.(0) * extents.(0)) + 1 |] st)
    (Array.append [| n |] extents)

(* The cases, which together reach each transpose fft.ml plans: lines off rows
   moved to and from the workspace, scattered, in columns of 40 and through a
   tail of four batch axes; the long lines of each transform, 7-smooth and
   Bluestein's, odd where real; long halved lines, whose split and pre-pass run
   on rows and on columns; and batches of three axes of odd extents that do not
   merge, whose lines' positions come from the tail by divisors, in a move of
   ragged tiles and in long lines' runs, splits and pre-passes. *)
let cases =
  [
    c2c "c2c of 300 scattered" (v [| 10; 2 |] [| 300; 5 |]);
    c2c "c2c of 300 flipped and scattered"
      (v ~offset:(299 * 10) [| -10; 2 |] [| 300; 5 |]);
    c2c "c2c of 3000 in columns of 40" (v [| 40; 1 |] [| 3000; 40 |]);
    c2c "c2c of 300 over four batch axes"
      (v [| 2; 172800; 28800; 7200; 1200 |] [| 300; 2; 3; 2; 3 |]);
    r2c "r2c of 64 scattered" (v [| 6; 2 |] [| 64; 3 |]);
    c2r "c2r of 64 scattered" ~n:64 (v [| 6; 2 |] [| 33; 3 |]);
    c2c "c2c of 8192" ~axes:[| 1 |] (packed [| 2; 8192 |]);
    r2c "r2c of 4375" ~axes:[| 1 |] (packed [| 2; 4375 |]);
    c2r "c2r of 4375" ~axes:[| 1 |] ~n:4375 (packed [| 2; 2188 |]);
    c2c "c2c of 2053" ~axes:[| 1 |] (packed [| 2; 2053 |]);
    r2c "r2c of 2053" ~axes:[| 1 |] (packed [| 2; 2053 |]);
    c2r "c2r of 2053" ~axes:[| 1 |] ~n:2053 (packed [| 2; (2053 / 2) + 1 |]);
    r2c "r2c of 16384" ~axes:[| 1 |] (packed [| 2; 16384 |]);
    c2r "c2r of 16384" ~axes:[| 1 |] ~n:16384 (packed [| 2; 8193 |]);
    r2c "r2c of 16384 in columns of 3" (v [| 3; 1 |] [| 16384; 3 |]);
    c2r "c2r of 16384 in columns of 3" ~n:16384 (v [| 3; 1 |] [| 8193; 3 |]);
    c2c "c2c of 300 scattered over [7; 11; 13]"
      (along_first 300 [| 7; 11; 13 |]);
    c2c "c2c of 8192 over [3; 5; 7]" ~axes:[| 3 |]
      (along_last 8192 [| 3; 5; 7 |]);
    r2c "r2c of 16384 over [3; 2; 5]" ~axes:[| 3 |]
      (along_last 16384 [| 3; 2; 5 |]);
    c2r "c2r of 16384 over [3; 2; 5]" ~axes:[| 3 |] ~n:16384
      (along_last 8193 [| 3; 2; 5 |]);
  ]

let widths c =
  match c.kind with `C2c -> (8, 8) | `R2c -> (4, 8) | `C2r -> (8, 4)

let span l width =
  let _, hi = L.span l in
  Int.max 16 (hi * width)

(* Runs *)

let code (t : F.work) =
  match t with
  | Transpose { input; scale; output; _ } ->
      Printf.sprintf "%c%c%c"
        (match input with Complex -> 'c' | Real -> 'r' | Bins -> 'b')
        (match scale with
        | Plain -> 'p'
        | Twiddle -> 't'
        | Chirp_in -> 'i'
        | Kernel -> 'k'
        | Chirp_out -> 'o')
        (match output with Complex -> 'c' | Half -> 'h' | Real -> 'r')
  | Split _ -> "split"
  | Prepass _ -> "prepass"
  | Step { step; _ } -> Printf.sprintf "s%d" step.l
  | Column_step { part = First_pass; _ } -> "c4096 first"
  | Column_step { part = Last_passes; _ } -> "c4096 last"

(* The works of [call], each its launches, in order. *)
let works (call : F.call) =
  Array.fold_right
    (fun (l : F.launch) acc ->
      match acc with
      | ((m : F.launch) :: _ as w) :: rest when m.work == l.work ->
          (l :: w) :: rest
      | _ -> [ l ] :: acc)
    call.launches []

let drawn st n =
  let b = Bytes.create n in
  for i = 0 to (n / 4) - 1 do
    put b (4 * i) (f32 (Random.State.float st 2. -. 1.))
  done;
  b

let device_of d bytes =
  let n = Bytes.length bytes in
  let buf = Rig.Buffer.create d n in
  Rig.Buffer.blit_from_string (Bytes.unsafe_to_string bytes) 0 buf 0 n;
  (buf, A.v D.Uint8 (L.contiguous [| n |]) buf)

let read_back buf =
  let n = Rig.Buffer.length buf in
  let b = Bytes.create n in
  Rig.Buffer.blit_to_bytes buf 0 b 0 n;
  b

(* A work's slots: its tables the only kept constants, and its sides' axes past
   what the words hold the call's own bytes, so that no layout evicts a
   table. *)
let check_slots (first : F.launch) (p : P.t) =
  let table key =
    List.exists
      (fun t -> String.starts_with ~prefix:("fft " ^ t ^ " ") key)
      [ "glo"; "ghi"; "tw"; "split"; "chirp"; "bluestein"; "rader"; "logs" ]
  in
  Array.iter
    (function
      | P.Constant { key; _ } ->
          equal ~msg:"a kept constant is a table" bool true (table key)
      | Operand _ | Workspace | Tickets | Fresh _ | Uploaded _ -> ())
    p.slots;
  let held (s : F.side) = Array.length s.dims <= held in
  equal ~msg:"the tail is the call's own" bool
    (not (held first.src && held first.dst))
    (Array.exists (function P.Uploaded _ -> true | _ -> false) p.slots)

(* Runs [w], a work's launches, on drawn bytes in every buffer, [fill] first
   writing into the source's: the source's bytes as run, the destination's
   before the run, and after it. *)
let execute ?(fill = ignore) c (call : F.call) ~inverse ~wide
    (w : F.launch list) =
  let d = S.device () in
  let st = Random.State.make [| 7 |] in
  let xw, dw = widths c in
  let sizes : F.buffer -> int = function
    | Operand -> span c.x xw
    | Dst -> span c.dst dw
    | Workspace _ -> Int.max 16 call.workspace
  in
  let key : F.buffer -> int = function
    | Operand -> 0
    | Dst -> 1
    | Workspace _ -> 2
  in
  let first = List.hd w and last = List.hd (List.rev w) in
  let bytes = Array.init 3 (fun _ -> None) in
  let drawn_of (s : F.buffer) =
    match bytes.(key s) with
    | Some b -> b
    | None ->
        let b = drawn st (sizes s) in
        bytes.(key s) <- Some b;
        b
  in
  let src_bytes = drawn_of first.src.buffer in
  fill src_bytes;
  let dst_bytes = Bytes.copy (drawn_of last.dst.buffer) in
  (* A column step's first pass writes the workspace its last passes read. *)
  List.iter (fun (l : F.launch) -> ignore (drawn_of l.dst.buffer)) w;
  let bufs = Array.map (Option.map (device_of d)) bytes in
  let on (s : F.buffer) = Option.get bufs.(key s) in
  let written =
    List.sort_uniq Int.compare
      (List.map (fun (l : F.launch) -> key l.dst.buffer) w)
  in
  let read_only =
    if List.mem (key first.src.buffer) written then []
    else [ key first.src.buffer ]
  in
  (* A buffer's slot is its operand's number in the call below: the written
     buffers, then the source where it is only read. *)
  let number k =
    let rec find i = function
      | [] -> invalid_arg "execute: a launch's buffer is no operand"
      | k' :: rest -> if k' = k then i else find (i + 1) rest
    in
    find 0 (written @ read_only)
  in
  let b = P.make () in
  let place (s : F.buffer) =
    let at = match s with Workspace at -> at | Operand | Dst -> 0 in
    (P.slot b (number (key s)), at)
  in
  List.iter (F.emit b ~wide ~inverse ~place) w;
  let p = P.finish b in
  check_slots first p;
  let read = List.map (fun k -> A.Any (snd (Option.get bufs.(k)))) read_only in
  let any k = A.Any (snd (Option.get bufs.(k))) in
  equal answer A.Done
    (Nx_metal.call d
       ~written:(Iarray.of_list (List.map any written))
       ~read:(Iarray.of_list read) p);
  (src_bytes, dst_bytes, read_back (fst (on last.dst.buffer)))

(* Special values, as float32 bits: zeros of both signs, the least and the
   greatest subnormals, the least normals, a normal whose products with a root
   fall below them, the greatest finites, infinities, and NaNs, one with a
   payload. *)
let specials =
  [|
    0x0000_0000l;
    0x8000_0000l;
    0x0000_0001l;
    0x807f_ffffl;
    0x0080_0000l;
    0x8080_0000l;
    0x0d80_0000l;
    0x7f7f_ffffl;
    0xff7f_ffffl;
    0x7f80_0000l;
    0xff80_0000l;
    0x7fc0_0000l;
    0xffc0_0123l;
  |]

(* Writes every special value into the first and the last line of [s], at its
   first points: each value in both of a complex point's parts, and at a real
   point, whatever the draw. *)
let put_specials (s : F.side) (input : F.input) b =
  let k = Array.length specials in
  List.iter
    (fun (line, shift) ->
      for i = 0 to (2 * k) - 1 do
        let v = specials.((i + shift) mod k) in
        let at =
          match input with
          | Real -> float_at s line i
          | Complex | Bins ->
              let r, im = parts s line (i / 2) in
              if i mod 2 = 0 then r else im
        in
        Bytes.set_int32_le b at v
      done)
    [ (0, 0); (lines s - 1, 1) ]

let nan_bits x = Int32.logand x 0x7fff_ffffl > 0x7f80_0000l

(* Whether two buffers hold the same floats: each word's bits, or a NaN in both,
   whose payload the GPU need not keep. *)
let same_floats a b =
  let n = Bytes.length a / 4 in
  let rec from i =
    i = n
    ||
    let x = Bytes.get_int32_le a (4 * i) and y = Bytes.get_int32_le b (4 * i) in
    (Int32.equal x y || (nan_bits x && nan_bits y)) && from (i + 1)
  in
  Bytes.length a = Bytes.length b && from 0

(* [got] is [expected] as [same_floats] says, or the check fails on their first
   differing word, printing both sides' bits, or on their lengths. *)
let same_bits ~msg expected got =
  if not (same_floats got expected) then begin
    let differs at =
      let x = Bytes.get_int32_le expected at
      and y = Bytes.get_int32_le got at in
      not (Int32.equal x y || (nan_bits x && nan_bits y))
    in
    let n = Int.min (Bytes.length got) (Bytes.length expected) in
    let at = ref 0 in
    while !at < n && not (differs !at) do
      at := !at + 4
    done;
    if !at >= n then
      equal ~msg:(msg ^ ": bytes") int (Bytes.length expected)
        (Bytes.length got)
    else
      equal
        ~msg:(Printf.sprintf "%s: the word at byte %d" msg !at)
        int32
        (Bytes.get_int32_le expected !at)
        (Bytes.get_int32_le got !at)
  end

(* Runs [w], a transpose's, a split's or a pre-pass's launches, on drawn bytes
   in every buffer, the special values among its source's, and checks every byte
   of the buffer it writes. *)
let run_work ?(fill = ignore) c (call : F.call) ~inverse ~wide
    (w : F.launch list) =
  let first = List.hd w in
  let input : F.input =
    match first.work with
    | Transpose { input; _ } -> input
    | Split _ | Prepass _ | Step _ | Column_step _ -> Complex
  in
  let fill b =
    put_specials first.src input b;
    fill b
  in
  let src, expected, got = execute ~fill c call ~inverse ~wide w in
  (match first.work with
  | Transpose _ -> transposed ~inverse first ~src ~dst:expected
  | Split _ | Prepass _ -> paired ~inverse first ~src ~dst:expected
  | Step _ | Column_step _ -> stepped ~inverse first ~src ~dst:expected);
  if not (same_floats got expected) then begin
    let at = ref 0 in
    let word_at i = Bytes.get_int32_le got i
    and model_at i = Bytes.get_int32_le expected i in
    while
      Int32.equal (word_at !at) (model_at !at)
      || (nan_bits (word_at !at) && nan_bits (model_at !at))
    do
      at := !at + 4
    done;
    let word = !at in
    failf "%s, %s%s%s: first differing word at byte %d: %h, expected %h" c.name
      (code first.work)
      (if inverse then ", inverse" else "")
      (if wide then ", 64-bit" else "")
      word (get got word) (get expected word)
  end

let reached = Hashtbl.create 16

(* The works with a side of more axes than their words hold, whose axes come
   from the tail and whose lines' positions take a division by an odd extent: a
   move, whose tile is ragged, a long line's runs, and a pair. *)
let tailed = Hashtbl.create 4

let divided (l : F.launch) =
  let past (s : F.side) =
    Array.length s.dims > held && Array.exists (fun e -> e mod 2 = 1) s.dims
  in
  past l.src || past l.dst

let all_tailed = [ "move"; "runs"; "pair" ]

let test_case c =
  let call = plan c in
  List.iter
    (fun (w : F.launch list) ->
      match (List.hd w).work with
      | Transpose { split = a, _; _ } as t ->
          Hashtbl.replace reached (code t) ();
          let first = List.hd w in
          if divided first then
            Hashtbl.replace tailed (if a = 1 then "move" else "runs") ();
          List.iter
            (fun wide ->
              List.iter
                (fun inverse -> run_work c call ~inverse ~wide w)
                [ false; true ])
            [ false; true ]
      | (Split _ | Prepass _) as t ->
          Hashtbl.replace reached (code t) ();
          if divided (List.hd w) then Hashtbl.replace tailed "pair" ();
          List.iter
            (fun wide ->
              List.iter
                (fun inverse -> run_work c call ~inverse ~wide w)
                [ false; true ])
            [ false; true ]
      | Step _ | Column_step _ -> ())
    (works call)

let all_moves =
  [
    "cpc";
    "rpr";
    "rpc";
    "bpc";
    "ctc";
    "cph";
    "cpr";
    "cic";
    "ric";
    "bic";
    "ckc";
    "coc";
    "coh";
    "cor";
  ]

let all_pairs = [ "split"; "prepass" ]

(* A completion's bin 0 (fft.mli, [Bins] and the direction): a long odd c2r's
   first launch, a [Bins] [Plain] transpose, writes point 0 of each line as [(Re
   X_0, +0)] forward and its conjugate [(Re X_0, -0)] in an inverse, whatever
   the sign of bin 0's imaginary part. *)
let test_bin0 () =
  let c = c2r "c2r of 4375" ~axes:[| 1 |] ~n:4375 (packed [| 2; 2188 |]) in
  let call = plan c in
  let w = List.hd (works call) in
  let first = List.hd w in
  equal ~msg:"the first launch" string "bpc" (code first.work);
  let re = 0.375 in
  let fill b =
    List.iter
      (fun (line, im) ->
        let pr, pi = parts first.src line 0 in
        put b pr re;
        put b pi im)
      [ (0, 0.); (1, -0.) ]
  in
  List.iter
    (fun (inverse, im) ->
      let _, _, got = execute ~fill c call ~inverse ~wide:false w in
      for line = 0 to 1 do
        let pr, pi = parts first.dst line 0 in
        let msg =
          Printf.sprintf "line %d%s" line (if inverse then ", inverse" else "")
        in
        equal ~msg int32 (Int32.bits_of_float re) (Bytes.get_int32_le got pr);
        equal ~msg int32 (Int32.bits_of_float im) (Bytes.get_int32_le got pi)
      done)
    [ (false, 0.); (true, -0.) ]

(* The cases of the step of 4096 points: lines packed, lines with gaps between
   them, and lines over three batch axes, past what the words hold. *)
let step_cases =
  [
    c2c "c2c of 4096" ~axes:[| 1 |] (packed [| 28; 4096 |]);
    c2c "c2c of 4096 over [4; 7]" ~axes:[| 2 |] (along_last 4096 [| 4; 7 |]);
    c2c "c2c of 4096 over [2; 7; 2]" ~axes:[| 3 |]
      (along_last 4096 [| 2; 7; 2 |]);
  ]

(* Line [line] of [s] a point [z] at [at] among zeros of drawn signs, so that
   its bins' parts are [z]'s products with roots and sums of exact zeros, whose
   signs and flushes the model fixes; of a float side, [z]'s real part. *)
let delta_at (s : F.side) b ~line ~at (zr, zi) =
  let st = Random.State.make [| 5; line |] in
  let zero () = if Random.State.bool st then 0. else -0. in
  for j = 0 to s.points - 1 do
    if s.width = 4 then
      put b (float_at s line j) (if j = at then zr else zero ())
    else begin
      let r, i = parts s line j in
      put b r (if j = at then zr else zero ());
      put b i (if j = at then zi else zero ())
    end
  done

(* The lines that hold special [i]: line 0 drawn, line 1 a delta of [(1, -0)],
   line [2 + i] special [i] in both parts of point 1, [(x, x)], and line [2 + k
   + i], [k] the specials, as [(x, -x)], whose sums cancel exactly; each among
   zeros, so that every root multiplies it. *)
let holders i =
  let k = Array.length specials in
  (2 + i, 2 + k + i)

let special_lines (s : F.side) b =
  let at = Int.min 1 (s.points - 1) in
  delta_at s b ~line:1 ~at:0 (1., -0.);
  Array.iteri
    (fun i v ->
      let x = Int32.float_of_bits v and same, opposite = holders i in
      delta_at s b ~line:same ~at (x, x);
      delta_at s b ~line:opposite ~at (x, -.x))
    specials

let finite x = Int32.logand x 0x7f80_0000l <> 0x7f80_0000l

(* The greatest finites, whose products with roots of both parts can overflow:
   the only finite specials whose results may hold no number. *)
let overflows v = Int32.logand v 0x7fff_ffffl = 0x7f7f_ffffl

(* The step of 4096 points writes its model's bits, in both directions: on drawn
   points, a delta of signed zeros, and each special value in lines of its own,
   each finite one's results finite, but the greatest finites', which overflow
   and yield a number at least. *)
let test_step () =
  List.iter
    (fun c ->
      let call = plan c in
      List.iter
        (fun (w : F.launch list) ->
          let first = List.hd w in
          equal ~msg:(c.name ^ ": one step") string "s4096" (code first.work);
          List.iter
            (fun inverse ->
              let src, expected, got =
                execute ~fill:(special_lines first.src) c call ~inverse
                  ~wide:false w
              in
              stepped ~inverse first ~src ~dst:expected;
              same_bits
                ~msg:(c.name ^ if inverse then ", inverse" else "")
                expected got;
              Array.iteri
                (fun i v ->
                  if finite v then
                    List.iter
                      (fun line ->
                        let finites = ref 0 in
                        for k = 0 to step_points - 1 do
                          let r, im = parts first.dst line k in
                          List.iter
                            (fun at ->
                              if finite (Bytes.get_int32_le expected at) then
                                incr finites)
                            [ r; im ]
                        done;
                        let msg =
                          Printf.sprintf "%s: line %d, special %lx's finites"
                            c.name line v
                        in
                        if overflows v then at_most ~msg int ~than:!finites 1
                        else equal ~msg int (2 * step_points) !finites)
                      (let same, opposite = holders i in
                       [ same; opposite ]))
                specials)
            [ false; true ])
        (works call))
    step_cases

(* The lengths of the specialised steps' cases: every radix, at 8 and 16 points
   a thread; one pass to seven; planes padded and not; one line a group to 256,
   groups ragged; and 288 threads a line. *)
(* Real lines: odd ones of one point, a pass of 3, 7^3 and 3^7, and halved ones
   of 1 to 4096 points, one point, one pass of 2, then of 8, an odd number of
   points (15 and 343), whose pairs include none with itself, odd radices, a
   power of two padded, and 8192's of 4096. A long halved line's steps are
   complex. *)
let real_lengths = [ 1; 2; 3; 4; 16; 30; 60; 343; 686; 1024; 2187; 8192 ]

(* Lines of a convolution in a group: Rader's of 11, 17 (16 points), 97 and
   2017, and Bluestein's of 22, 23, 1031 (m 2304) and 2039 (m 4096); real ones,
   odd of 11 and 23, and halved of 22, 2062 and 4078, h of 11, 1031 and 2039. *)
let conv_lengths = [ 11; 17; 22; 23; 97; 1031; 2017; 2039 ]
let real_conv_lengths = [ 11; 22; 23; 2062; 4078 ]

let key_lengths =
  [
    1;
    2;
    3;
    4;
    5;
    7;
    8;
    16;
    60;
    128;
    343;
    1000;
    1024;
    1458;
    2048;
    2187;
    2401;
    3675;
  ]

(* Lengths whose passes take forms the others leave, on both paths: a first pass
   of 3 at 8 points a thread (9), one between others (72) and a last of 4 (32);
   passes of 2, 4 and 8 between others at 16 (1440, 1600, 1920); a last of 5 at
   16 (1400), and of 7 side by side (1372); first passes at 16 of 2, 4, 5, 7 and
   8 (1470, 1500, 1875, 1575, 1400). *)
let form_lengths =
  [ 9; 32; 72; 1372; 1400; 1440; 1470; 1500; 1575; 1600; 1875; 1920 ]

let side_of : F.path -> string = function
  | Contiguous -> "rows "
  | Side_by_side -> "side "

(* The unrolled forms of a step's passes on [path]: each radix at its points a
   thread, as a first pass, which loads its points, a last, which stores them,
   or one between. *)
let pass_forms (step : F.step) path =
  let p = step.p in
  let last = Array.length step.radices - 1 in
  List.mapi
    (fun i r ->
      Printf.sprintf "%sradix %d at %d%s%s" (side_of path) r p
        (if i = 0 then ", first" else "")
        (if i = last then ", last" else ""))
    (Array.to_list step.radices)

(* Every pass form a specialised step takes: those of each 7-smooth length other
   than 4096, on rows, and side by side to [F.side_most], by its side step. *)
let reachable_forms () =
  let forms = Hashtbl.create 64 in
  for l = 2 to F.most_points - 1 do
    if (F.line l ~real:false).how = Smooth then begin
      let step = F.step l in
      List.iter
        (fun f -> Hashtbl.replace forms f ())
        (pass_forms step Contiguous);
      if l <= F.side_most then
        List.iter
          (fun f -> Hashtbl.replace forms f ())
          (pass_forms (F.side_step l) Side_by_side)
    end
  done;
  List.sort compare (List.of_seq (Hashtbl.to_seq_keys forms))

(* A specialised step writes its model's bits, in both directions: on drawn
   points, a delta of signed zeros, and each special value in lines of its own,
   28 lines a call, rows and, up to 2048 points, columns, which run side by side
   where the plan takes them so; and calls of fewer lines than a group holds,
   whose groups take the least power of two of slots at least their lines: 3 on
   4, one empty, and 1 on 1. *)
let test_key_step () =
  let kinds = Hashtbl.create 16 and forms = Hashtbl.create 64 in
  let check c =
    let call = plan c in
    let keyed_works =
      List.filter (fun w -> Option.is_some (F.key (List.hd w))) (works call)
    in
    if keyed_works = [] then fail (c.name ^ ": no specialised step");
    List.iter
      (fun w ->
        let first : F.launch = List.hd w in
        let step, path =
          match first.work with
          | Step { step; path; _ } -> (step, path)
          | _ -> fail (c.name ^ ": a key off a step")
        in
        List.iter (fun f -> Hashtbl.replace forms f ()) (pass_forms step path);
        (match F.key first with
        | Some (_, cs) ->
            let open Nx_metal.Kernels in
            let conv =
              if cs.(fft_fc_conv) = fft_conv_rader then "rader "
              else if cs.(fft_fc_conv) = fft_conv_bluestein then "bluestein "
              else ""
            in
            List.iter
              (fun k ->
                let kind = side_of path ^ String.trim (conv ^ k) in
                Hashtbl.replace kinds kind ())
              [
                (if cs.(fft_fc_halved) = fft_halved_split then "split" else "");
                (if cs.(fft_fc_halved) = fft_halved_prepass then "pre-pass"
                 else "");
                (if cs.(fft_fc_input) = fft_in_real then "reals in" else "");
                (if cs.(fft_fc_input) = fft_in_bins then "bins in" else "");
                (if cs.(fft_fc_input) = fft_in_none then "nothing in" else "");
                (if cs.(fft_fc_output) = fft_out_half then "half out" else "");
                (if cs.(fft_fc_output) = fft_out_real then "reals out" else "");
              ]
        | None -> ());
        List.iter
          (fun inverse ->
            let src, expected, got =
              execute ~fill:(special_lines first.src) c call ~inverse
                ~wide:false w
            in
            keyed ~inverse first step ~src ~dst:expected;
            same_bits
              ~msg:(c.name ^ if inverse then ", inverse" else "")
              expected got)
          [ false; true ])
      keyed_works
  in
  let bins n = (n / 2) + 1 in
  let name kind shape = Printf.sprintf "%s of %s" kind shape in
  (* Long convolutions' models cost most, and so do long lines kept for their
     passes' forms: 4 lines of each, which their columns' calls route side by
     side, where 3 take the transposes. *)
  let count n = if n > 1000 && not (List.mem n key_lengths) then 4 else 28 in
  let across n = Printf.sprintf "%d x %d" (count n) n
  and down n = Printf.sprintf "%d x %d" n (count n) in
  let complex n =
    let c = count n in
    let rows = c2c (name "c2c" (across n)) ~axes:[| 1 |] (packed [| c; n |]) in
    if n > F.side_most then [ rows ]
    else [ rows; c2c (name "c2c" (down n)) (packed [| n; c |]) ]
  in
  let real n =
    let c = count n in
    [
      r2c (name "r2c" (across n)) ~axes:[| 1 |] (packed [| c; n |]);
      c2r (name "c2r" (across n)) ~axes:[| 1 |] ~n (packed [| c; bins n |]);
    ]
    @
    if n <= F.side_most then
      [
        r2c (name "r2c" (down n)) (packed [| n; c |]);
        c2r (name "c2r" (down n)) ~n (packed [| bins n; c |]);
      ]
    else []
  in
  let few =
    [
      (c2c "c2c of 3 x 64" ~axes:[| 1 |] (packed [| 3; 64 |]), F.Contiguous, 4);
      (c2c "c2c of 1 x 64" ~axes:[| 1 |] (packed [| 1; 64 |]), F.Contiguous, 1);
      (c2c "c2c of 64 x 3" (packed [| 64; 3 |]), F.Side_by_side, 4);
    ]
  in
  let cases =
    List.concat_map complex (key_lengths @ form_lengths @ conv_lengths)
    @ List.concat_map real (real_lengths @ real_conv_lengths)
    @ [ r2c "r2c of 28 x 0" ~axes:[| 1 |] (packed [| 28; 0 |]) ]
    @ List.map (fun (c, _, _) -> c) few
  in
  (* Metal makes a call's new pipelines together: the cases' keys at once, in
     place of a call's few at a time. *)
  Nx_metal.prepare (S.device ()) (List.map plan cases);
  List.iter check cases;
  List.iter
    (fun (c, path, lines) ->
      match (plan c).launches.(0).work with
      | Step s ->
          equal ~msg:(c.name ^ ": path") string (side_of path) (side_of s.path);
          equal ~msg:(c.name ^ ": lines a group") int lines s.lines
      | _ -> fail (c.name ^ ": no step"))
    few;
  let kinds_of path =
    List.map
      (fun k -> side_of path ^ k)
      ([
         "";
         "bins in";
         "bluestein";
         "bluestein bins in";
         "bluestein half out";
         "bluestein pre-pass";
         "bluestein reals in";
         "bluestein reals out";
         "bluestein split";
         "half out";
         "pre-pass";
         "rader";
         "rader bins in";
         "rader half out";
         "rader reals in";
         "rader reals out";
         "reals in";
         "reals out";
         "split";
       ]
      @ match path with F.Contiguous -> [ "nothing in" ] | Side_by_side -> [])
  in
  equal ~msg:"the kinds of each path" (list string)
    (List.sort compare (kinds_of Contiguous @ kinds_of Side_by_side))
    (List.sort compare (List.of_seq (Hashtbl.to_seq_keys kinds)));
  equal ~msg:"pass forms no length reached" (list string) []
    (List.filter (fun f -> not (Hashtbl.mem forms f)) (reachable_forms ()))

(* The cases of the column step: 4096 by 40 packed, two blocks of 16 lines and
   one of 8, and its lines flipped; two runs of 20 lines along axis 1; and 2 by
   3 runs of 17 lines that do not merge, whose sides' axes come from the
   tail. *)
let column_cases =
  [
    c2c "c2c of 4096 x 40" (packed [| 4096; 40 |]);
    c2c "c2c of 4096 x 40 flipped" (v ~offset:39 [| 40; -1 |] [| 4096; 40 |]);
    c2c "c2c of [2; 4096; 20]" ~axes:[| 1 |] (packed [| 2; 4096; 20 |]);
    c2c "c2c of [2; 3; 4096; 17]" ~axes:[| 2 |]
      (v
         [| (3 * ((4096 * 17) + 1)) + 1; (4096 * 17) + 1; 17; 1 |]
         [| 2; 3; 4096; 17 |]);
  ]

(* A column step's first pass and last passes write each line the step's bits,
   in both directions, on drawn points, a delta of signed zeros, and each
   special value in lines of its own. *)
let test_column_step () =
  List.iter
    (fun c ->
      let call = plan c in
      match works call with
      | [ (first :: _ as fw); (last :: _ as lw) ] ->
          equal ~msg:(c.name ^ ": the works") (list string)
            [ "c4096 first"; "c4096 last" ]
            [ code first.work; code last.work ];
          List.iter
            (fun inverse ->
              let src, expected, got =
                execute ~fill:(special_lines first.src) c call ~inverse
                  ~wide:false (fw @ lw)
              in
              stepped_on ~inverse first.src last.dst ~src ~dst:expected;
              same_bits
                ~msg:(c.name ^ if inverse then ", inverse" else "")
                expected got)
            [ false; true ]
      | _ -> fail (c.name ^ ": not a column step's two works"))
    column_cases

(* References, in double *)

(* The forward DFT of the complex points [x] of [n] parts pairs at bins [0 <= k
   < count]: [Σ_j x_j w_n^jk], [w_n = e^(-2πi/n)]. *)
let dft (x : (float * float) array) count =
  let n = Array.length x in
  let c =
    Array.init n (fun e ->
        cos (-2. *. Float.pi *. Float.of_int e /. Float.of_int n))
  and s =
    Array.init n (fun e ->
        sin (-2. *. Float.pi *. Float.of_int e /. Float.of_int n))
  in
  Array.init count (fun k ->
      let re = ref 0. and im = ref 0. and e = ref 0 and k = k mod n in
      for j = 0 to n - 1 do
        let xr, xi = x.(j) in
        re := !re +. (xr *. c.(!e)) -. (xi *. s.(!e));
        im := !im +. (xr *. s.(!e)) +. (xi *. c.(!e));
        e := if !e + k >= n then !e + k - n else !e + k
      done;
      (!re, !im))

let u = Float.ldexp 1. (-24)
let modulus (r, i) = Float.hypot r i

(* A long halved line's split and pre-pass against the definitions of their
   results, on [n = 2 h] real points [x] drawn a line. A split takes [Z = DFT_h
   (y)], [y_j = x_2j + i x_2j+1], rounded to float32, into [X = DFT_n (x)], bins
   [k <= h]; a pre-pass takes [X] rounded into the [Z] whose inverse DFT,
   unnormalised, is the c2r of [X]'s pairs, [2 DFT_h (y)]. Each is within [16 u
   (|a| + |b|)] of it at pair [(k, h - k)], [a] and [b] its operands there and
   [u = 2^-24]: the operands' rounding and the root's, and about a rounding of
   each step. *)

(* The cases draw their lines from one seed, so they share lines: each line's
   two DFTs, [Z] and [X], are computed once. *)
let references = Hashtbl.create 8

let reference x h =
  match Hashtbl.find_opt references x with
  | Some r -> r
  | None ->
      let pairs = Array.init h (fun j -> (x.(2 * j), x.((2 * j) + 1))) in
      let r = (dft pairs h, dft (Array.map (fun v -> (v, 0.)) x) (h + 1)) in
      Hashtbl.add references x r;
      r

let check_defined c =
  let call = plan c in
  let st = Random.State.make [| 11 |] in
  let checked = ref false in
  List.iter
    (fun (w : F.launch list) ->
      let first = List.hd w in
      match first.work with
      | Split _ | Prepass _ ->
          checked := true;
          let split = match first.work with Split _ -> true | _ -> false in
          let bins = if split then first.dst else first.src in
          let h = bins.points - 1 in
          let lines = lines bins in
          let xs =
            Array.init lines (fun _ ->
                Array.init (2 * h) (fun _ ->
                    f32 (Random.State.float st 2. -. 1.)))
          in
          let zs = Array.map (fun x -> fst (reference x h)) xs
          and xbins = Array.map (fun x -> snd (reference x h)) xs in
          let operands = if split then zs else xbins in
          let fill b =
            Array.iteri
              (fun l values ->
                Array.iteri
                  (fun k (r, i) ->
                    let pr, pi = parts first.src l k in
                    put b pr (f32 r);
                    put b pi (f32 i))
                  values)
              operands
          in
          let _, _, got =
            execute ~fill c call ~inverse:(not split) ~wide:false w
          in
          let out = if split then h + 1 else h in
          for l = 0 to lines - 1 do
            let expected =
              if split then xbins.(l)
              else Array.map (fun (r, i) -> (2. *. r, 2. *. i)) zs.(l)
            in
            for k = 0 to out - 1 do
              let pr, pi = parts first.dst l k in
              let gr = get got pr and gi = get got pi in
              let er, ei = expected.(k) in
              let k' = Int.min k (h - k) in
              (* A split's [Z_h] is [Z_0]. *)
              let other = if split && k' = 0 then 0 else h - k' in
              let a = operands.(l).(k') and b = operands.(l).(other) in
              let bound = 16. *. u *. (modulus a +. modulus b) in
              at_most
                ~msg:
                  (Printf.sprintf "%s, %s: line %d, point %d" c.name
                     (code first.work) l k)
                float_exact ~than:bound
                (modulus (gr -. er, gi -. ei))
            done
          done
      | Transpose _ | Step _ | Column_step _ -> ())
    (works call);
  equal ~msg:(c.name ^ ": a split or a pre-pass") bool true !checked

let defined_cases =
  [
    r2c "r2c of 8232" ~axes:[| 1 |] (packed [| 2; 8232 |]);
    r2c "r2c of 8232 in columns of 3" (v [| 3; 1 |] [| 8232; 3 |]);
    c2r "c2r of 8232" ~axes:[| 1 |] ~n:8232 (packed [| 2; 4117 |]);
    c2r "c2r of 8232 in columns of 3" ~n:8232 (v [| 3; 1 |] [| 4117; 3 |]);
  ]

let test_defined () = List.iter check_defined defined_cases

(* A long line's steps are FFTs alone: a long Bluestein line's, as in a c2c of
   2049 points, take keys of no convolution; its steps of 4096 points, as a c2c
   of [4096²]'s, take fft_s4096. *)
let test_unrun_step () =
  let steps c =
    List.filter
      (fun (l : F.launch) -> match l.work with Step _ -> true | _ -> false)
      (Array.to_list (plan c).launches)
  in
  let emit (l : F.launch) =
    F.emit (P.make ()) ~inverse:false ~place:(fun _ -> (0, 0)) l
  in
  let bluestein =
    steps (c2c "c2c of 2049" ~axes:[| 1 |] (packed [| 1; 2049 |]))
  in
  at_least ~msg:"a long Bluestein line's steps" int ~than:4
    (List.length bluestein);
  List.iter
    (fun (l : F.launch) ->
      let conv =
        match F.key l with
        | Some (_, cs) -> cs.(Nx_metal.Kernels.fft_fc_conv)
        | None -> Nx_metal.Kernels.fft_conv_none
      in
      equal ~msg:"a long Bluestein line's step" int
        Nx_metal.Kernels.fft_conv_none conv;
      emit l)
    bluestein;
  let long =
    steps (c2c "c2c of 4096²" ~axes:[| 1 |] (packed [| 1; 4096 * 4096 |]))
  in
  equal ~msg:"a long line's steps" (list string) [ "s4096"; "s4096" ]
    (List.map (fun (l : F.launch) -> code l.work) long);
  List.iter
    (fun (l : F.launch) ->
      equal ~msg:"a long line's key" (option string) None
        (Option.map (fun (k, cs) -> P.label k cs) (F.key l)))
    long;
  List.iter emit long

(* A delta [a] at point [j], its spectrum's point [k] [a w_n^(jk)], in
   double. *)
let delta_dft (ar, ai) j n =
  Array.init n (fun k ->
      let e = j * k mod n in
      let t = -2. *. Float.pi *. Float.of_int e /. Float.of_int n in
      let c = cos t and s = sin t in
      ((ar *. c) -. (ai *. s), (ar *. s) +. (ai *. c)))

let delta_case =
  let open Gen in
  with_pp
    (fun ppf (j, (ar, ai)) -> Format.fprintf ppf "(%h, %h) at %d" ar ai j)
    (pair
       (frequency
          [
            (1, constant 0);
            (1, constant (step_points / 2));
            (1, map (fun k -> 1 lsl k) (int_range 0 11));
            (4, int_range 0 (step_points - 1));
          ])
       (pair (map f32 (float_range (-1.) 1.)) (map f32 (float_range (-1.) 1.))))

(* Deltas that every run checks: at the line's edges and its middle, and at [273
   r = 256 r + 16 r + r], [r < 16], each digit of the three passes [r], so that
   every butterfly position is reached. *)
let delta_edges =
  List.map
    (fun j -> (j, (f32 0.75, f32 (-0.5))))
    ([ 1; step_points / 2; step_points - 1 ] @ List.init 16 (fun r -> 273 * r))

(* The step's model is the DFT within the bound fft.mli states, [e(n) u √n ‖x‖₂]
   at [e(n) = 5.35 log₂ n], on deltas, whose spectra double gives in [O(n)]. *)
let law_step_dft (j, a) =
  cover "the delta at point 0" (j = 0);
  cover "the delta at an odd point" (j mod 2 = 1);
  let x = Array.make step_points (0., 0.) in
  x.(j) <- a;
  let y = s4096 ~inverse:false x and exact = delta_dft a j step_points in
  let norm v =
    sqrt (Array.fold_left (fun s z -> s +. (modulus z ** 2.)) 0. v)
  in
  let err = norm (Array.map2 (fun (a, b) (c, d) -> (a -. c, b -. d)) y exact) in
  let n = Float.of_int step_points in
  at_most ~msg:"the error's norm" float_exact
    ~than:(5.35 *. Float.log2 n *. u *. sqrt n *. modulus a)
    err

let smooth_lengths =
  let rec smooth n = function
    | [] -> n = 1
    | p :: ps -> if n mod p = 0 then smooth (n / p) (p :: ps) else smooth n ps
  in
  List.filter (fun n -> smooth n [ 2; 3; 5; 7 ]) (List.init 4096 succ)

let key_delta_case =
  let open Gen in
  with_pp
    (fun ppf (n, j, (ar, ai)) ->
      Format.fprintf ppf "(%h, %h) at %d of %d" ar ai j n)
    (let* n = of_list ~pp:Format.pp_print_int smooth_lengths in
     let* j =
       frequency
         [ (1, constant 0); (1, constant (n - 1)); (4, int_range 0 (n - 1)) ]
     in
     let+ a =
       pair (map f32 (float_range (-1.) 1.)) (map f32 (float_range (-1.) 1.))
     in
     (n, j, a))

(* A delta at the last point of lines whose passes take every radix: 2187's
   seven of 3, 3675's of 3, 5 and 7, 1000's of 8 and 5, 1600's of 16 and 4,
   1458's of 2 and 3. *)
let key_delta_edges =
  List.map
    (fun n -> (n, n - 1, (f32 0.75, f32 (-0.5))))
    [ 2187; 3675; 1000; 1600; 1458 ]

(* A specialised step's model is the DFT within the bound fft.mli states at
   every 7-smooth length, on deltas. *)
let law_key_dft (n, j, a) =
  let step = F.step n in
  List.iter
    (fun r ->
      cover (Printf.sprintf "a pass of %d" r) (Array.mem r step.radices))
    [ 2; 3; 4; 5; 7; 8; 16 ];
  let x = Array.make n (0., 0.) in
  x.(j) <- a;
  let y = key_fft ~inverse:false step x and exact = delta_dft a j n in
  let norm v =
    sqrt (Array.fold_left (fun s z -> s +. (modulus z ** 2.)) 0. v)
  in
  let err = norm (Array.map2 (fun (a, b) (c, d) -> (a -. c, b -. d)) y exact) in
  let n = Float.of_int n in
  at_most ~msg:"the error's norm" float_exact
    ~than:(5.35 *. Float.log2 n *. u *. sqrt n *. modulus a)
    err

(* Calls *)

(* Runs every launch of [call] on [operand], the destination and the workspace
   zeroed first: the destination's bytes after the run. *)
let run_call c (call : F.call) ~inverse operand =
  let d = S.device () in
  let _, dw = widths c in
  let bufs =
    [|
      device_of d operand;
      device_of d (Bytes.make (span c.dst dw) '\000');
      device_of d (Bytes.make (Int.max 16 call.workspace) '\000');
    |]
  in
  let b = P.make () in
  (* A buffer's slot is its operand's number in the call below. *)
  let place (s : F.buffer) =
    let at = match s with Workspace at -> at | Operand | Dst -> 0 in
    let number = match s with Dst -> 0 | Workspace _ -> 1 | Operand -> 2 in
    (P.slot b number, at)
  in
  Array.iter (F.emit b ~inverse ~place) call.launches;
  let any i = A.Any (snd bufs.(i)) in
  equal answer A.Done
    (Nx_metal.call d
       ~written:(Iarray.of_list [ any 1; any 2 ])
       ~read:(Iarray.of_list [ any 0 ])
       (P.finish b));
  read_back (fst bufs.(1))

(* A buffer of complex points with each imaginary part's sign flipped. *)
let conj_bytes b =
  let b = Bytes.copy b in
  for i = 0 to (Bytes.length b / 8) - 1 do
    let at = (8 * i) + 4 in
    Bytes.set_int32_le b at
      (Int32.logxor (Bytes.get_int32_le b at) Int32.min_int)
  done;
  b

(* The paths a c2c runs on, those whose steps have kernels: a line in a group,
   and a line of columns, through transposes to and from the workspace. The law
   skips the paths whose steps have none yet: long lines, Rader's, and
   Bluestein's in a group and long, and the c2r's, odd against its completed c2c
   and halved. *)
let call_cases =
  [
    c2c "c2c of 4096" ~axes:[| 1 |] (packed [| 28; 4096 |]);
    c2c "c2c of 4096 in columns of 28" (v [| 28; 1 |] [| 4096; 28 |]);
  ]

(* An inverse c2c is the conjugate of the forward transform of its conjugated
   operand, bit for bit but for a NaN's payload (fft.mli, calls): on drawn
   points, a delta of signed zeros, and each special value in a line of its
   own. *)
let test_conj_law () =
  List.iter
    (fun c ->
      let call = plan c in
      let first = call.launches.(0) in
      let st = Random.State.make [| 17 |] in
      let x = drawn st (span c.x 8) in
      special_lines first.src x;
      let inverse = run_call c call ~inverse:true x in
      let forward =
        conj_bytes (run_call c call ~inverse:false (conj_bytes x))
      in
      same_bits ~msg:(c.name ^ ": the conjugated forward") forward inverse)
    call_cases

(* Nx_metal.c2c *)

let c2c_shape = [| 256; 4096 |]

(* [x]'s points, in C order, as a complex64 array of [c2c_shape] on [d]: packed,
   [offset] elements into its buffer, at byte [at] of a view of a buffer, so
   that its first point lies [at] past a line, or, where [columns], each line a
   column, its points [rows] apart, so that the call moves them through the
   workspace. *)
let c2c_operand ?(columns = false) d x ~at ~offset =
  let rows = c2c_shape.(0) and n = c2c_shape.(1) in
  let x =
    if not columns then x
    else
      let t = Bytes.create (Bytes.length x) in
      for i = 0 to rows - 1 do
        for j = 0 to n - 1 do
          Bytes.blit x (8 * ((i * n) + j)) t (8 * ((j * rows) + i)) 8
        done
      done;
      t
  in
  let bytes = Bytes.length x in
  let buf = Rig.Buffer.create d (at + (8 * offset) + bytes) in
  Rig.Buffer.blit_from_string (Bytes.unsafe_to_string x) 0 buf
    (at + (8 * offset))
    bytes;
  let b =
    if at = 0 then buf
    else Rig.Buffer.view buf ~first:at ~length:((8 * offset) + bytes)
  in
  let strides = if columns then [| 1; rows |] else [| n; 1 |] in
  A.v D.Complex64 (v ~offset strides c2c_shape) b

(* Nx_metal.c2c runs a complex64 call whose every launch has a kernel, with the
   bits of its plan's launches, in both directions, from a packed operand, one
   at an element offset, one at a byte of a line, one whose first point starts
   between two complex elements, on byte 4 of 8, and one in columns; it declines
   a call with a step it has no kernel for, and a complex128 one. *)
let test_c2c () =
  let d = S.device () in
  let c = c2c "c2c of 256 x 4096" ~axes:[| 1 |] (packed c2c_shape) in
  let call = plan c in
  let x = drawn (Random.State.make [| 23 |]) (8 * 256 * 4096) in
  let axes = Iarray.of_list [ 1 ] in
  List.iter
    (fun (direction : Nx_kernel.direction) ->
      let inverse = direction = Inverse in
      let expected = run_call c call ~inverse x in
      List.iter
        (fun (name, columns, at, offset) ->
          let dst = A.create d D.Complex64 (Iarray.of_array c2c_shape) in
          let msg =
            Printf.sprintf "%s%s" name (if inverse then ", inverse" else "")
          in
          equal ~msg answer A.Done
            (Nx_metal.c2c direction ~axes ~dst
               (c2c_operand ~columns d x ~at ~offset));
          same_bits ~msg expected (read_back (A.buffer dst)))
        [
          ("packed", false, 0, 0);
          ("3 elements in", false, 0, 3);
          ("at byte 120", false, 120, 0);
          ("at byte 124", false, 124, 0);
          ("in columns", true, 0, 0);
        ])
    [ Forward; Inverse ];
  let long = c2c "c2c of 4 x 8192" ~axes:[| 1 |] (packed [| 4; 8192 |]) in
  let x = drawn (Random.State.make [| 31 |]) (8 * 4 * 8192) in
  let buf, _ = device_of d x in
  let dst = A.create d D.Complex64 (Iarray.of_list [ 4; 8192 ]) in
  equal ~msg:"8192 points" answer A.Done
    (Nx_metal.c2c Forward ~axes ~dst
       (A.v D.Complex64 (packed [| 4; 8192 |]) buf));
  same_bits ~msg:"8192 points"
    (run_call long (plan long) ~inverse:false x)
    (read_back (A.buffer dst));
  let real = r2c "r2c of 4 x 1000" ~axes:[| 1 |] (packed [| 4; 1000 |]) in
  let x = drawn (Random.State.make [| 37 |]) (4 * 4 * 1000) in
  let buf, _ = device_of d x in
  let dst = A.create d D.Complex64 (Iarray.of_list [ 4; 501 ]) in
  equal ~msg:"r2c" answer A.Done
    (Nx_metal.r2c ~axes ~dst (A.v D.Float32 (packed [| 4; 1000 |]) buf));
  same_bits ~msg:"r2c"
    (run_call real (plan real) ~inverse:false x)
    (read_back (A.buffer dst));
  let bins =
    c2r "c2r of 4 x 1000" ~axes:[| 1 |] ~n:1000 (packed [| 4; 501 |])
  in
  let x = drawn (Random.State.make [| 41 |]) (8 * 4 * 501) in
  let buf, _ = device_of d x in
  let dst = A.create d D.Float32 (Iarray.of_list [ 4; 1000 ]) in
  equal ~msg:"c2r" answer A.Done
    (Nx_metal.c2r ~axes ~dst (A.v D.Complex64 (packed [| 4; 501 |]) buf));
  same_bits ~msg:"c2r"
    (run_call bins (plan bins) ~inverse:true x)
    (read_back (A.buffer dst));
  let wide = A.create d D.Complex128 (Iarray.of_array c2c_shape) in
  equal ~msg:"complex128" answer A.Declined
    (Nx_metal.c2c Forward ~axes
       ~dst:(A.create d D.Complex128 (Iarray.of_array c2c_shape))
       wide)

(* Wide sides *)

(* The bound a test lowers [wide] to: the span of a step's rows, a halved line's
   8192 floats, the least nx_metal.mli allows. *)
let wide_least = (2 * step_points) - 1
let span_most = (1 lsl 31) - 1

let bounded n f =
  Nx_metal.bound_fft_wide n;
  Fun.protect ~finally:(fun () -> Nx_metal.bound_fft_wide span_most) f

(* The codes of [call]'s works of 64-bit offsets. *)
let wide_codes (call : F.call) =
  List.filter_map
    (fun (w : F.launch list) ->
      let t = (List.hd w).work in
      match t with
      | Transpose { wide = true; _ }
      | Split { wide = true }
      | Prepass { wide = true } ->
          Some (code t)
      | _ -> None)
    (works call)

type wcase = { kind : [ `C2c | `R2c | `C2r ]; axis : int; shape : int array }

let pp_wcase ppf c =
  Format.fprintf ppf "%s along %d of %s"
    (match c.kind with `C2c -> "c2c" | `R2c -> "r2c" | `C2r -> "c2r")
    c.axis
    (String.concat "x" (Array.to_list (Array.map string_of_int c.shape)))

(* The operand's and the result's shapes: [shape] is the complex or real side's,
   the bins [n / 2 + 1] along [axis]. *)
let wcase_shapes c =
  let bins = Array.copy c.shape in
  bins.(c.axis) <- (c.shape.(c.axis) / 2) + 1;
  match c.kind with
  | `C2c -> (c.shape, c.shape)
  | `R2c -> (c.shape, bins)
  | `C2r -> (bins, c.shape)

let wcase_plan c =
  let x, y = wcase_shapes c in
  let f = match c.kind with `C2c -> F.c2c | `R2c -> F.r2c | `C2r -> F.c2r in
  Option.get (f ~bases:(0, 0) ~axes:[| c.axis |] (packed x) ~dst:(packed y))

(* [c]'s call on the GPU from [x], the operand's bytes: its result's bytes. *)
let wcase_run d c x =
  let xs, ys = wcase_shapes c in
  let axes = Iarray.of_list [ c.axis ] in
  let buf, _ = device_of d x in
  let msg = Format.asprintf "%a: answer" pp_wcase c in
  match c.kind with
  | `C2c ->
      let dst = A.create d D.Complex64 (Iarray.of_array ys) in
      equal ~msg answer A.Done
        (Nx_metal.c2c Forward ~axes ~dst (A.v D.Complex64 (packed xs) buf));
      read_back (A.buffer dst)
  | `R2c ->
      let dst = A.create d D.Complex64 (Iarray.of_array ys) in
      equal ~msg answer A.Done
        (Nx_metal.r2c ~axes ~dst (A.v D.Float32 (packed xs) buf));
      read_back (A.buffer dst)
  | `C2r ->
      let dst = A.create d D.Float32 (Iarray.of_array ys) in
      equal ~msg answer A.Done
        (Nx_metal.c2r ~axes ~dst (A.v D.Complex64 (packed xs) buf));
      read_back (A.buffer dst)

(* Lines past [wide_least]'s span, along rows and in columns, of each class: a
   short smooth line in columns, a long one, a long Bluestein one, a short
   Bluestein and a Rader line in columns; real ones halved, odd and in columns,
   and an r2c's real side one float past the bound. *)
let wcases =
  [
    { kind = `C2c; axis = 0; shape = [| 64; 256 |] };
    { kind = `C2c; axis = 1; shape = [| 2; 16384 |] };
    { kind = `C2c; axis = 1; shape = [| 2; 8209 |] };
    { kind = `C2c; axis = 0; shape = [| 2053; 8 |] };
    { kind = `C2c; axis = 0; shape = [| 4001; 4 |] };
    { kind = `R2c; axis = 1; shape = [| 2; 16384 |] };
    { kind = `R2c; axis = 0; shape = [| 256; 64 |] };
    { kind = `R2c; axis = 1; shape = [| 2; 8194 |] };
    { kind = `R2c; axis = 1; shape = [| 2; 8209 |] };
    { kind = `R2c; axis = 0; shape = [| 10125; 2 |] };
    { kind = `C2r; axis = 1; shape = [| 2; 16384 |] };
    { kind = `C2r; axis = 0; shape = [| 256; 64 |] };
    { kind = `C2r; axis = 1; shape = [| 2; 8209 |] };
    { kind = `C2r; axis = 1; shape = [| 2; 10125 |] };
  ]

(* A halved line's real rows at the bound: its step's side stays narrow. *)
let wbounds =
  [
    { kind = `R2c; axis = 1; shape = [| 2; 8192 |] };
    { kind = `C2r; axis = 1; shape = [| 2; 8192 |] };
  ]

let wide_reached = Hashtbl.create 16

(* A call whose sides are wide under a lowered bound, where [wide], plans its
   lines around the steps, through the 64-bit transposes and pairs, and computes
   the bits it computes under the library's bound, where they are narrow. *)
let wide_case ~wide c =
  let d = S.device () in
  let xs, _ = wcase_shapes c in
  let point = match c.kind with `R2c -> 4 | `C2c | `C2r -> 8 in
  let x =
    drawn
      (Random.State.make [| Hashtbl.hash c |])
      (point * Array.fold_left ( * ) 1 xs)
  in
  let msg m = Format.asprintf "%a: %s" pp_wcase c m in
  equal
    ~msg:(msg "wide works at the library's bound")
    (list string) []
    (wide_codes (wcase_plan c));
  let narrow = wcase_run d c x in
  bounded wide_least (fun () ->
      let codes = wide_codes (wcase_plan c) in
      List.iter (fun k -> Hashtbl.replace wide_reached k ()) codes;
      if wide then
        not_equal
          ~msg:(msg "wide works at the lowered bound")
          (list string) [] codes
      else equal ~msg:(msg "wide works at the bound") (list string) [] codes;
      same_bits ~msg:(msg "bits") narrow (wcase_run d c x))

(* [wide_case] of each case, and every 64-bit transpose and pair reached. *)
let test_wide () =
  List.iter (wide_case ~wide:true) wcases;
  List.iter (wide_case ~wide:false) wbounds;
  let all = all_moves @ all_pairs in
  equal ~msg:"64-bit works reached" (list string) all
    (List.filter (Hashtbl.mem wide_reached) all)

(* The bound holds a step's rows narrow and stays within 32-bit offsets. *)
let test_wide_bound () =
  raises_match Exn.invalid_arg (fun () ->
      Nx_metal.bound_fft_wide (wide_least - 1));
  raises_match Exn.invalid_arg (fun () ->
      Nx_metal.bound_fft_wide (span_most + 1))

(* A c2c of 4096 x 4096 along both axes, whose second axis runs on [dst] through
   the workspace, has the bits of its plan's launches in both directions. *)
let test_c2c2 () =
  let d = S.device () in
  let shape = [| step_points; step_points |] in
  let c = c2c "c2c of 4096 x 4096" ~axes:[| 0; 1 |] (packed shape) in
  let call = plan c in
  let x = drawn (Random.State.make [| 29 |]) (8 * step_points * step_points) in
  let buf, _ = device_of d x in
  let operand = A.v D.Complex64 (packed shape) buf in
  List.iter
    (fun (direction : Nx_kernel.direction) ->
      let inverse = direction = Inverse in
      let msg = if inverse then "inverse" else "forward" in
      let expected = run_call c call ~inverse x in
      let dst = A.create d D.Complex64 (Iarray.of_array shape) in
      equal ~msg answer A.Done
        (Nx_metal.c2c direction ~axes:(Iarray.of_list [ 0; 1 ]) ~dst operand);
      same_bits ~msg expected (read_back (A.buffer dst)))
    [ Forward; Inverse ]

(* A mixed-radix line of 6000 points and a Bluestein line of 10007 each pass
   their points through workspace from one launch to the next. Run a part a
   launch, so that a part ends between those launches, Nx_metal.c2c writes the
   bits of one submission of the plan. *)
let test_passes_parted () =
  let d = S.device () in
  let handoffs (call : F.call) =
    let l = call.launches and n = ref 0 in
    for i = 1 to Array.length l - 1 do
      match (l.(i - 1).dst.buffer, l.(i).src.buffer) with
      | Workspace _, Workspace _ -> incr n
      | _ -> ()
    done;
    !n
  in
  List.iter
    (fun n ->
      let shape = [| 3; n |] in
      let name = Printf.sprintf "c2c of 3 x %d" n in
      let c = c2c name ~axes:[| 1 |] (packed shape) in
      let call = plan c in
      greater ~msg:(name ^ ": handoffs") int ~than:0 (handoffs call);
      let x = drawn (Random.State.make [| n |]) (8 * 3 * n) in
      let expected = run_call c call ~inverse:false x in
      let buf, _ = device_of d x in
      let dst = A.create d D.Complex64 (Iarray.of_array shape) in
      S.parted ~launches:1 (fun () ->
          equal ~msg:name answer A.Done
            (Nx_metal.c2c Forward ~axes:(Iarray.of_list [ 1 ]) ~dst
               (A.v D.Complex64 (packed shape) buf)));
      same_bits ~msg:name expected (read_back (A.buffer dst)))
    [ 6000; 10007 ]

(* A dense line's transform in double, [O(n²)], from roots of [n] each within an
   ulp of a double. *)
let dft ~inverse (x : (float * float) array) =
  let n = Array.length x in
  let sign = if inverse then 1. else -1. in
  let w =
    Array.init n (fun m ->
        let t = 2. *. Float.pi *. Float.of_int m /. Float.of_int n in
        (Float.cos t, sign *. Float.sin t))
  in
  Array.init n (fun k ->
      let re = ref 0. and im = ref 0. in
      for j = 0 to n - 1 do
        let xr, xi = x.(j) and wr, wi = w.(j * k mod n) in
        re := !re +. ((xr *. wr) -. (xi *. wi));
        im := !im +. ((xr *. wi) +. (xi *. wr))
      done;
      (!re, !im))

(* Nx_metal.c2c of [lines] lines of 4096 points [xs], each a pair of float32
   values, in C order: its result's bytes. *)
let transformed ~inverse ~lines xs =
  let d = S.device () and n = step_points in
  let bytes = Bytes.create (8 * lines * n) in
  Array.iteri
    (fun i (r, im) ->
      put bytes (8 * i) r;
      put bytes ((8 * i) + 4) im)
    xs;
  let buf, _ = device_of d bytes in
  let shape = [| lines; n |] in
  let dst = A.create d D.Complex64 (Iarray.of_array shape) in
  let direction : Nx_kernel.direction = if inverse then Inverse else Forward in
  equal ~msg:"answer" answer A.Done
    (Nx_metal.c2c direction ~axes:(Iarray.of_list [ 1 ]) ~dst
       (A.v D.Complex64 (packed shape) buf));
  read_back (A.buffer dst)

(* Fails unless each line of [got], the transform of [xs], lies within the bound
   nx_metal.mli states of the line's transform in double: [5.35 log₂ n u √n ‖x‖₂
   + c n 2^-126], the second term the flushing of subnormals. *)
let within_bound ~inverse ~lines xs got =
  let n = step_points in
  for l = 0 to lines - 1 do
    let x = Array.sub xs (l * n) n in
    let exact = dft ~inverse x in
    let err = ref 0. and norm = ref 0. in
    for k = 0 to n - 1 do
      let at = 8 * ((l * n) + k) in
      let er, ei = exact.(k) in
      err :=
        !err +. ((get got at -. er) ** 2.) +. ((get got (at + 4) -. ei) ** 2.);
      norm := !norm +. (modulus x.(k) ** 2.)
    done;
    let fn = Float.of_int n in
    at_most
      ~msg:(Printf.sprintf "line %d's error" l)
      float_exact
      ~than:
        ((5.35 *. Float.log2 fn *. u *. sqrt fn *. sqrt !norm)
        +. (F.flush_scale *. fn *. Float.ldexp 1. (-126)))
      (sqrt !err)
  done

(* Two lines of 4096 points: dense, each part [±m 2^(e - k)], [m] drawn in [0.5,
   1] and [k] in [0, spread], or where [tone] is [Some k0] the tone [2^e e^(2πi
   j k0 / n)], whose bins but one cancel. Parts lie below [2^115], so that
   nothing overflows, and down to [2^-150], so that loads, products and
   cancelling sums flush. *)
type dense = {
  seed : int;
  e : int;
  spread : int;
  tone : int option;
  inverse : bool;
}

let dense_gen =
  let open Gen in
  with_pp
    (fun ppf c ->
      Format.fprintf ppf "seed %d, 2^%d, spread %d%s%s" c.seed c.e c.spread
        (match c.tone with
        | Some k -> Printf.sprintf ", tone %d" k
        | None -> "")
        (if c.inverse then ", inverse" else ""))
    (bind
       (frequency [ (1, constant 0); (1, constant 20) ])
       (fun spread ->
         map
           (fun ((seed, e), (tone, inverse)) ->
             { seed; e; spread; tone; inverse })
           (pair
              (pair int
                 (frequency
                    [
                      (1, int_range (spread - 150) (spread - 90));
                      (2, int_range (spread - 150) 115);
                      (1, int_range 105 115);
                    ]))
              (pair
                 (frequency
                    [
                      (3, constant None);
                      (1, map Option.some (int_range 0 (step_points - 1)));
                    ])
                 bool))))

let dense_examples =
  let case ?(spread = 0) ?tone ?(inverse = false) seed e =
    { seed; e; spread; tone; inverse }
  in
  [
    case 1 0;
    case 2 (-80) ~spread:20 ~inverse:true;
    case 3 115;
    case 4 115 ~spread:20 ~inverse:true;
    case 5 (-100) ~tone:1;
    case 7 (-120) ~spread:20;
    case 8 (-140) ~tone:7 ~inverse:true;
    case 6 115 ~tone:1365 ~inverse:true;
  ]

(* Nx_metal.c2c's dense lines and tones lie within the bound, from subnormal
   parts to parts near [2^115]. *)
let law_dense c =
  cover "a line of one magnitude" (c.spread = 0 && c.tone = None);
  cover "a line of spread magnitudes" (c.spread > 0 && c.tone = None);
  cover "a tone" (c.tone <> None);
  cover "parts near 2^115" (c.e >= 110);
  cover "parts near 2^-100" (c.e - c.spread <= -95);
  cover "subnormal parts" (c.e - c.spread < -126);
  let n = step_points and lines = 2 in
  let st = Random.State.make [| c.seed |] in
  let part () =
    let m = 0.5 +. Random.State.float st 0.5 in
    let m = if Random.State.bool st then m else -.m in
    f32 (Float.ldexp m (c.e - Random.State.int st (c.spread + 1)))
  in
  let point i =
    match c.tone with
    | None -> (part (), part ())
    | Some k0 ->
        let t =
          2. *. Float.pi *. Float.of_int (i mod n * k0 mod n) /. Float.of_int n
        in
        ( f32 (Float.ldexp (Float.cos t) c.e),
          f32 (Float.ldexp (Float.sin t) c.e) )
  in
  let xs = Array.init (lines * n) point in
  within_bound ~inverse:c.inverse ~lines xs
    (transformed ~inverse:c.inverse ~lines xs)

(* The line of parts [±m] whose forward bin 1 has the greatest real part, [m Σ_j
   (|cos θ_j| + |sin θ_j|)], about [1.27 · 4096 m], [θ_j = 2π j / n]. *)
let greatest m =
  Array.init step_points (fun j ->
      let t = 2. *. Float.pi *. Float.of_int j /. Float.of_int step_points in
      let sign v = if v < 0. then -.m else m in
      (sign (Float.cos t), sign (Float.sin t)))

(* A line whose parts lie below [(1 - 2^-17) (π/2) 2^115] overflows nowhere: the
   line whose bin 1 is greatest lies within the bound at [2^115] and at the
   greatest float32 below that threshold, its bin 1 then about [2^-17]
   relatively below [2^128]. The same line at [1.6875 · 2^115] has one exact
   part past [2^128], bin 1's real part: that part is an infinity of its sign,
   and every part below [2^128] by more than the bound is finite. *)
let test_overflow () =
  let at = Float.ldexp 1. 115 in
  let inside =
    let v = (1. -. Float.ldexp 1. (-17)) *. (Float.pi /. 2.) *. at in
    let r = f32 v in
    if r < v then r
    else Int32.float_of_bits (Int32.pred (Int32.bits_of_float r))
  in
  List.iter
    (fun m ->
      let xs = greatest m in
      let got = transformed ~inverse:false ~lines:1 xs in
      within_bound ~inverse:false ~lines:1 xs got)
    [ at; inside ];
  let xs = greatest (1.6875 *. at) in
  let past = transformed ~inverse:false ~lines:1 xs in
  let exact = dft ~inverse:false xs in
  let limit = Float.ldexp 1. 128 and slack = Float.ldexp 1. (-10) in
  let overflowed = ref 0 in
  Array.iteri
    (fun k (er, ei) ->
      List.iteri
        (fun p v ->
          let got = get past ((8 * k) + (4 * p)) in
          let msg = Printf.sprintf "bin %d's part %d, exactly %h" k p v in
          if Float.abs v > limit *. (1. +. slack) then begin
            incr overflowed;
            equal ~msg float_exact (Float.copy_sign infinity v) got
          end
          else if Float.abs v < limit *. (1. -. slack) then
            equal ~msg bool true (Float.is_finite got))
        [ er; ei ])
    exact;
  equal ~msg:"parts past 2^128" int 1 !overflowed

(* The greatest float32 below [v]. *)
let below v =
  let r = f32 v in
  if r < v then r else Int32.float_of_bits (Int32.pred (Int32.bits_of_float r))

(* Bin 0 of an r2c of [n] points of one value [v]: [n v], which an r2c's halved
   line sums as [E.re + U = 2 n v] before halving. *)
let constant_r2c n v =
  let d = S.device () in
  let bytes = Bytes.create (4 * n) in
  for j = 0 to n - 1 do
    put bytes (4 * j) v
  done;
  let buf, _ = device_of d bytes in
  let dst = A.create d D.Complex64 (Iarray.of_list [ 1; (n / 2) + 1 ]) in
  equal ~msg:"answer" answer A.Done
    (Nx_metal.r2c ~axes:(Iarray.of_list [ 1 ]) ~dst
       (A.v D.Float32 (packed [| 1; n |]) buf));
  let got = read_back (A.buffer dst) in
  (get got 0, get got 4)

(* Bin 0 of a c2c of [n] points of one value [v + i v]: [n (v + i v)]. *)
let constant_c2c n v =
  let d = S.device () in
  let bytes = Bytes.create (8 * n) in
  for j = 0 to (2 * n) - 1 do
    put bytes (4 * j) v
  done;
  let buf, _ = device_of d bytes in
  let dst = A.create d D.Complex64 (Iarray.of_list [ 1; n ]) in
  equal ~msg:"answer" answer A.Done
    (Nx_metal.c2c Forward ~axes:(Iarray.of_list [ 1 ]) ~dst
       (A.v D.Complex64 (packed [| 1; n |]) buf));
  let got = read_back (A.buffer dst) in
  (get got 0, get got 4)

(* A real line or one past 4096 points overflows nowhere below its threshold,
   [(1 - 2^-17) 2^124 / n]: an r2c of 4096 points of one value and a c2c of
   8192, at the threshold, compute bin 0 exactly. An r2c's split sums [2 n v],
   past [2^128] at the threshold of shorter complex lines, which a test saw
   overflow. *)
let test_real_overflow () =
  let at n = below ((1. -. Float.ldexp 1. (-17)) *. Float.ldexp 1. 124 /. n) in
  let v = at 4096. in
  equal ~msg:"an r2c's bin 0"
    (pair float_exact float_exact)
    (4096. *. v, 0.)
    (constant_r2c 4096 v);
  let v = at 8192. in
  equal ~msg:"a long c2c's bin 0"
    (pair float_exact float_exact)
    (8192. *. v, 8192. *. v)
    (constant_c2c 8192 v)

(* The float32 parts of an operand of [count] elements of [width] parts, each
   [v], on the device. *)
let constant_on d ~count ~width v =
  let bytes = Bytes.create (4 * width * count) in
  for j = 0 to (width * count) - 1 do
    put bytes (4 * j) v
  done;
  fst (device_of d bytes)

(* Fails unless every float of [got] is finite and its first lies within [e
   2^-24 √n ‖x‖₂ + c n 2^-126] of [exact], [norm] being [‖x‖₂]. *)
let finite_within ~msg ~e ~c ~n ~norm exact got =
  for at = 0 to (Bytes.length got / 4) - 1 do
    if not (finite (Bytes.get_int32_le got (4 * at))) then
      fail (Printf.sprintf "%s: float %d is %h" msg at (get got (4 * at)))
  done;
  let fn = Float.of_int n in
  let bound =
    (e *. Float.ldexp 1. (-24) *. Float.sqrt fn *. norm)
    +. (c *. fn *. Float.ldexp 1. (-126))
  in
  less ~msg float_exact ~than:bound (Float.abs (get got 0 -. exact))

(* The key caches against a model, under {!Nx_metal.bound_keys} [b]: a device
   keeps the functions of the [b] keys made last; a call whose submission it
   keeps makes nothing, else it makes the keys it does not keep; and it keeps
   the submissions of such calls newest first while they launch at most [b] keys
   together. Each call is a c2c of a line of a length of [cache_lengths], into
   another buffer or in place: one key, which the two share, and two
   submissions, as their operands take different slots. Its operands are kept
   for it, so that its submission can be found, and its bits are those of its
   length's first call without a bound, whether its key was kept, found from the
   other's, or dropped and made again. *)
let cache_bound = 3
let cache_lengths = [| 2; 3; 4; 5; 6; 8 |]

type key_model = {
  mutable keys : int list;
  mutable calls : (int * int) list;
  mutable called : int list;
}

(* The first [k] of a list, its newest. *)
let rec newest k = function
  | [] -> []
  | x :: rest -> if k <= 0 then [] else x :: newest (k - 1) rest

(* Each call's operands, a line of points [j + 1 + i j] written before each
   call, and its first call's bits, made without a bound, by length, out of
   place and in place. *)
let cache_operands =
  lazy
    (let d = S.device () in
     Array.map
       (fun n ->
         let points = Bytes.create (8 * n) in
         for j = 0 to n - 1 do
           put points (8 * j) (Float.of_int (j + 1));
           put points ((8 * j) + 4) (Float.of_int j)
         done;
         let points = Bytes.to_string points in
         Array.map
           (fun in_place ->
             let buf = Rig.Buffer.create d (8 * n) in
             let x = A.v D.Complex64 (packed [| 1; n |]) buf in
             let dst =
               if in_place then x
               else A.create d D.Complex64 (Iarray.of_list [ 1; n ])
             in
             let run () =
               Rig.Buffer.blit_from_string points 0 buf 0 (8 * n);
               equal ~msg:"answer" answer A.Done
                 (Nx_metal.c2c Forward ~axes:(Iarray.of_list [ 1 ]) ~dst x);
               Bytes.to_string (read_back (A.buffer dst))
             in
             (run, run ()))
           [| false; true |])
       cache_lengths)

let key_caches =
  let device = abstract "d" in
  (* A run of calls, so that keys drop between forgets: up to 6 calls, which
     reach a dropped key in about 4 programs of 10. *)
  let which =
    Gen.list ~size:(Gen.int_range 1 6)
      (Gen.with_pp
         (fun ppf (i, l) ->
           Format.fprintf ppf "%d %s" cache_lengths.(i)
             (if l = 0 then "out" else "in place"))
         (Gen.pair
            (Gen.int_range 0 (Array.length cache_lengths - 1))
            (Gen.int_range 0 1)))
  in
  (* One device, so every handle shares one model, which a forget empties. *)
  let model = { keys = []; calls = []; called = [] } in
  let forget () =
    model.keys <- [];
    model.calls <- [];
    model.called <- [];
    model
  in
  let call1 m (i, l) =
    let n = cache_lengths.(i) in
    let kept_call = List.mem (n, l) m.calls and kept_key = List.mem n m.keys in
    cover "a key another call made, found" ((not kept_call) && kept_key);
    cover "a dropped key made again"
      ((not kept_call) && (not kept_key) && List.mem n m.called);
    m.called <- n :: m.called;
    if not kept_call then begin
      if not kept_key then m.keys <- newest cache_bound (n :: m.keys);
      m.calls <- newest cache_bound ((n, l) :: m.calls)
    end;
    ( List.length m.keys,
      List.length m.calls,
      snd (Lazy.force cache_operands).(i).(l) )
  in
  let forget_on () =
    let d = S.device () in
    ignore (Lazy.force cache_operands);
    Nx_metal.forget_keys d;
    d
  in
  let call calls m = List.map (call1 m) calls in
  let call_on1 d (i, l) =
    let run, _ = (Lazy.force cache_operands).(i).(l) in
    Nx_metal.bound_keys cache_bound;
    let bits = Fun.protect ~finally:(fun () -> Nx_metal.bound_keys 256) run in
    (Nx_metal.kept_keys_on d, Nx_metal.kept_call_keys_on d, bits)
  in
  let call_on calls d = List.map (call_on1 d) calls in
  stateful ~count:40 ~steps:16
    "the key caches keep the keys and calls made last"
    [
      command "forget" (Gen.unit @-> makes device) forget forget_on;
      command "call"
        (which @-> device ^-> returns (list (triple int int string)))
        call call_on;
    ]

(* Values that only a convolution or a pre-pass computes stay finite under their
   lines' thresholds, the least that apply, as nx_metal.mli argues them. Lines
   of one value at the threshold, [v + i v] where complex: a c2r of 8192 points,
   all of whose bins its pre-pass makes from [v + i v] (bins 0 and 4096 read as
   [v]); Bluestein's line of 1031 points in a c2c, and halved, of 2062 points,
   in an r2c and a c2r. Each output is finite and its first, the exact sum [n
   v], or [n (v + i v)] for the c2c, lies within the bound. *)
let test_convolution_overflow () =
  let d = S.device () in
  let at m = below ((1. -. Float.ldexp 1. (-17)) *. Float.ldexp 1. 124 /. m) in
  let log2 x = Float.log2 (Float.of_int x) in
  let blue n =
    match (F.line n ~real:false).how with
    | Bluestein m -> m
    | Smooth | Rader -> fail (Printf.sprintf "%d is no Bluestein line" n)
  in
  let r2c n v =
    let x = constant_on d ~count:n ~width:1 v in
    let dst = A.create d D.Complex64 (Iarray.of_list [ 1; (n / 2) + 1 ]) in
    equal ~msg:"answer" answer A.Done
      (Nx_metal.r2c ~axes:(Iarray.of_list [ 1 ]) ~dst
         (A.v D.Float32 (packed [| 1; n |]) x));
    read_back (A.buffer dst)
  in
  let c2r n v =
    let bins = (n / 2) + 1 in
    let x = constant_on d ~count:bins ~width:2 v in
    let dst = A.create d D.Float32 (Iarray.of_list [ 1; n ]) in
    equal ~msg:"answer" answer A.Done
      (Nx_metal.c2r ~axes:(Iarray.of_list [ 1 ]) ~dst
         (A.v D.Complex64 (packed [| 1; bins |]) x));
    read_back (A.buffer dst)
  in
  let c2c n v =
    let x = constant_on d ~count:n ~width:2 v in
    let dst = A.create d D.Complex64 (Iarray.of_list [ 1; n ]) in
    equal ~msg:"answer" answer A.Done
      (Nx_metal.c2c Forward ~axes:(Iarray.of_list [ 1 ]) ~dst
         (A.v D.Complex64 (packed [| 1; n |]) x));
    read_back (A.buffer dst)
  in
  (* A c2r's completed bins: [v] at 0 and [n / 2], [v ± i v] between. *)
  let c2r_norm n v = v *. Float.sqrt (Float.of_int ((2 * n) - 2)) in
  let n = 8192 in
  let v = at (Float.of_int n) in
  finite_within ~msg:"a c2r of 8192 through its pre-pass"
    ~e:((5.35 *. log2 (n / 2)) +. 20.)
    ~c:F.flush_scale ~n ~norm:(c2r_norm n v)
    (Float.of_int n *. v)
    (c2r n v);
  let blue_e n = (91. *. log2 n) +. 236. in
  let n = 1031 in
  let m = blue n in
  let v = at (Float.of_int (m * m)) in
  finite_within ~msg:"Bluestein's c2c of 1031" ~e:(blue_e n)
    ~c:(F.bluestein_flush_scale n)
    ~n
    ~norm:(v *. Float.sqrt (Float.of_int (2 * n)))
    (Float.of_int n *. v)
    (c2c n v);
  let n = 2062 and h = 1031 in
  let m = blue h in
  let v = at (Float.of_int (m * m)) in
  finite_within ~msg:"Bluestein's halved r2c of 2062"
    ~e:(blue_e h +. 11.)
    ~c:(F.bluestein_flush_scale h)
    ~n
    ~norm:(v *. Float.sqrt (Float.of_int n))
    (Float.of_int n *. v)
    (r2c n v);
  finite_within ~msg:"Bluestein's halved c2r of 2062"
    ~e:(blue_e h +. 20.)
    ~c:(F.bluestein_flush_scale h)
    ~n ~norm:(c2r_norm n v)
    (Float.of_int n *. v)
    (c2r n v)

(* An operand of a c2c along [axis], of 4096 points, its other axes a batch:
   laid out in memory from the axis [order.(0)] out, each axis's stride [gap]
   times the span of those inside it, negated where [neg], or 0 where [bcast];
   [offset] elements past its first element, at byte [at] of a view of its
   buffer. Where [pitch > 0], the axis's stride is [pitch] in place of the
   span. *)
type layout_case = {
  shape : int array;
  axis : int;
  order : int array;
  gap : int array;
  neg : bool array;
  bcast : bool array;
  offset : int;
  at : int;
  inverse : bool;
  pitch : int;
}

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat ";" (Array.to_list (Array.map string_of_int a)))

let pp_bools ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat ";"
       (Array.to_list (Array.map (fun b -> if b then "t" else "f") a)))

let pp_layout_case ppf c =
  Format.fprintf ppf
    "shape %a axis %d order %a gap %a neg %a bcast %a offset %d at %d pitch \
     %d%s"
    pp_ints c.shape c.axis pp_ints c.order pp_ints c.gap pp_bools c.neg pp_bools
    c.bcast c.offset c.at c.pitch
    (if c.inverse then " inverse" else "")

(* The case's strides and the element its index 0 lies at, past every negative
   stride's reach. *)
let case_strides c =
  let r = Array.length c.shape in
  let strides = Array.make r 0 and run = ref 1 in
  Array.iter
    (fun a ->
      if not c.bcast.(a) then begin
        let step = if a = c.axis && c.pitch > 0 then c.pitch else !run in
        strides.(a) <- (if c.neg.(a) then -step else step);
        run := step * c.shape.(a) * c.gap.(a)
      end)
    c.order;
  let first = ref c.offset and last = ref c.offset in
  Array.iteri
    (fun a s ->
      let reach = (c.shape.(a) - 1) * abs s in
      if s < 0 then first := !first + reach;
      last := !last + reach)
    strides;
  (strides, !first, !last + 1)

(* A case's batch: the indices of its lines in C order, each the element its
   point 0 lies at past [first]. *)
let line_starts c strides =
  let r = Array.length c.shape in
  let starts = ref [ 0 ] in
  for a = r - 1 downto 0 do
    if a <> c.axis then
      starts :=
        List.concat_map
          (fun i -> List.map (fun s -> (i * strides.(a)) + s) !starts)
          (List.init c.shape.(a) Fun.id)
  done;
  !starts

(* A columns axis for the column step: 4096 by 16 to 40 lines one element apart,
   packed or a line's points 4 KiB, 8 KiB, 8 KiB and 8 bytes, or 16 KiB apart,
   the points or the lines flipped. *)
let columns_gen =
  let open Gen in
  map
    (fun ((k, pitch), (flip, at, inverse)) ->
      {
        shape = [| step_points; k |];
        axis = 0;
        order = [| 1; 0 |];
        gap = [| 1; 1 |];
        neg = [| flip = 1; flip = 2 |];
        bcast = [| false; false |];
        offset = 0;
        at;
        inverse;
        pitch;
      })
    (pair
       (pair (int_range 16 40) (of_list [ 0; 512; 1024; 1025; 2048 ]))
       (triple (int_range 0 3) (of_list [ 0; 120; 124 ]) bool))

let layout_gen =
  let open Gen in
  with_pp pp_layout_case
    (frequency
       [
         (1, columns_gen);
         ( 3,
           bind
             (frequency
                [ (1, constant 0); (2, constant 1); (3, int_range 2 3) ])
             (fun batch ->
               let r = batch + 1 in
               bind (int_range 0 batch) (fun axis ->
                   bind
                     (array ~size:(constant batch) (int_range 1 3))
                     (fun ext ->
                       let shape =
                         Array.init r (fun i ->
                             if i < axis then ext.(i)
                             else if i = axis then step_points
                             else ext.(i - 1))
                       in
                       let flags p = array ~size:(constant r) (frequency p) in
                       map
                         (fun ( (order, gap, neg),
                                (bcast, line_bcast),
                                (offset, at, inverse) ) ->
                           let bcast =
                             Array.mapi
                               (fun i b -> if i = axis then line_bcast else b)
                               bcast
                           in
                           {
                             shape;
                             axis;
                             order = Array.of_list order;
                             gap;
                             neg;
                             bcast;
                             offset;
                             at;
                             inverse;
                             pitch = 0;
                           })
                         (triple
                            (triple
                               (permutation (List.init r Fun.id))
                               (flags [ (2, constant 1); (1, constant 2) ])
                               (flags
                                  [ (3, constant false); (1, constant true) ]))
                            (pair
                               (flags
                                  [ (3, constant false); (1, constant true) ])
                               (frequency
                                  [ (7, constant false); (1, constant true) ]))
                            (triple
                               (frequency
                                  [ (1, constant 0); (1, int_range 1 5) ])
                               (frequency
                                  [
                                    (2, constant 0);
                                    (1, constant 120);
                                    (1, map (fun k -> 8 * k) (int_range 1 15));
                                    ( 1,
                                      map
                                        (fun k -> (8 * k) + 4)
                                        (int_range 0 15) );
                                  ])
                               bool))))) );
       ])

(* Cases that every run checks, one of each class [law_layout] covers. *)
let layout_examples =
  let case ?(gap = [||]) ?(neg = [||]) ?(bcast = [||]) ?(offset = 0) ?(at = 0)
      ?(inverse = false) ?(pitch = 0) shape axis order =
    let r = Array.length shape in
    let fill a v = if a = [||] then Array.make r v else a in
    {
      shape;
      axis;
      order;
      gap = fill gap 1;
      neg = fill neg false;
      bcast = fill bcast false;
      offset;
      at;
      inverse;
      pitch;
    }
  in
  let n = step_points in
  [
    case [| n |] 0 [| 0 |];
    case [| 3; n |] 1 [| 1; 0 |] ~gap:[| 2; 1 |] ~offset:3 ~inverse:true;
    case [| 2; n; 3 |] 1 [| 2; 1; 0 |] ~neg:[| true; false; true |];
    case [| 3; n |] 1 [| 1; 0 |] ~bcast:[| true; false |] ~at:120;
    case [| 2; n |] 1 [| 1; 0 |] ~bcast:[| false; true |];
    case [| 2; n |] 1 [| 1; 0 |] ~neg:[| false; true |] ~at:8;
    case [| 2; n |] 1 [| 1; 0 |] ~at:124;
    case [| 2; 2; 2; n |] 3 [| 3; 2; 1; 0 |] ~gap:[| 1; 2; 2; 2 |];
    case [| 3; 2; n; 3 |] 2 [| 0; 3; 1; 2 |] ~inverse:true;
    case [| n; 16 |] 0 [| 1; 0 |];
    case [| n; 17 |] 0 [| 1; 0 |] ~at:124 ~inverse:true;
    case [| n; 33 |] 0 [| 1; 0 |] ~at:120;
    case [| n; 20 |] 0 [| 1; 0 |] ~pitch:1024 ~neg:[| true; false |];
    case [| n; 20 |] 0 [| 1; 0 |] ~pitch:1025;
  ]

(* A c2c of a case's operand writes each line the bits a packed run of its
   points gives, so that a line's bits depend on its points alone. Where
   [parted], the calls run a part a launch. *)
let law_layout ?(parted = false) c =
  let d = S.device () in
  let strides, first, elements = case_strides c in
  let st = Random.State.make [| Hashtbl.hash c |] in
  let starts = line_starts c strides in
  let ps = strides.(c.axis) in
  let mem =
    if c.pitch = 0 then drawn st (8 * elements)
    else begin
      (* A pitch's span is mostly gaps: only the lines' points are drawn. *)
      let b = Bytes.make (8 * elements) '\000' in
      List.iter
        (fun s ->
          for j = 0 to step_points - 1 do
            let at = 8 * (first + s + (j * ps)) in
            put b at (f32 (Random.State.float st 2. -. 1.));
            put b (at + 4) (f32 (Random.State.float st 2. -. 1.))
          done)
        starts;
      b
    end
  in
  let x = v ~offset:first strides c.shape in
  let call =
    F.c2c ~bases:(c.at mod P.line, 0) ~axes:[| c.axis |] x ~dst:(packed c.shape)
  in
  let launches = match call with Some c -> c.launches | None -> [||] in
  let tailed (s : F.side) = Array.length s.dims > held in
  (* A part a launch: a call of several launches is several parts. *)
  if parted then cover "a call of several parts" (Array.length launches > 1);
  cover "a gapped stride"
    (Array.exists2 (fun g b -> g = 2 && not b) c.gap c.bcast);
  cover "a negative stride" (Array.exists (fun s -> s < 0) strides);
  cover "a broadcast batch axis"
    (Array.exists
       (fun i -> i <> c.axis && c.bcast.(i) && c.shape.(i) > 1)
       (Array.init (Array.length c.shape) Fun.id));
  cover "a broadcast line" c.bcast.(c.axis);
  cover "an element offset" (c.offset > 0);
  cover "a view off a line" (c.at mod P.line <> 0);
  cover "a start between two complex elements" (c.at mod 8 = 4);
  cover "the axis not last" (c.axis < Array.length c.shape - 1);
  cover "lines through workspace"
    (Array.exists
       (fun (l : F.launch) ->
         match l.work with Transpose _ -> true | _ -> false)
       launches);
  cover "a side's axes past the words"
    (Array.exists (fun (l : F.launch) -> tailed l.src || tailed l.dst) launches);
  cover "the column step"
    (Array.exists
       (fun (l : F.launch) ->
         match l.work with Column_step _ -> true | _ -> false)
       launches);
  cover "16 lines past 8 KiB apart through workspace"
    (c.pitch > 1024
    && Array.exists
         (fun (l : F.launch) ->
           match l.work with Transpose _ -> true | _ -> false)
         launches);
  let buf = Rig.Buffer.create d (c.at + Bytes.length mem) in
  Rig.Buffer.blit_from_string
    (Bytes.unsafe_to_string mem)
    0 buf c.at (Bytes.length mem);
  let operand =
    A.v D.Complex64 x
      (if c.at = 0 then buf
       else Rig.Buffer.view buf ~first:c.at ~length:(Bytes.length mem))
  in
  let lines = List.length starts in
  let rows = Bytes.create (8 * lines * step_points) in
  List.iteri
    (fun l s ->
      for j = 0 to step_points - 1 do
        Bytes.blit mem
          (8 * (first + s + (j * ps)))
          rows
          (8 * ((l * step_points) + j))
          8
      done)
    starts;
  let direction : Nx_kernel.direction =
    if c.inverse then Inverse else Forward
  in
  let run ~axis x =
    let dst = A.create d D.Complex64 (L.shape (A.layout x)) in
    equal ~msg:"answer" answer A.Done
      (Nx_metal.c2c direction ~axes:(Iarray.of_list [ axis ]) ~dst x);
    read_back (A.buffer dst)
  in
  let rows_buf, _ = device_of d rows in
  let by_rows =
    run ~axis:1 (A.v D.Complex64 (packed [| lines; step_points |]) rows_buf)
  in
  let got = run ~axis:c.axis operand in
  (* The result's lines in C order of the batch, as the packed run lays them
     out. *)
  let dst_strides = Iarray.to_array (L.strides (packed c.shape)) in
  let ds = dst_strides.(c.axis) in
  let dst_starts = line_starts c dst_strides in
  let lined = Bytes.create (Bytes.length by_rows) in
  List.iteri
    (fun l s ->
      for j = 0 to step_points - 1 do
        Bytes.blit got
          (8 * (s + (j * ds)))
          lined
          (8 * ((l * step_points) + j))
          8
      done)
    dst_starts;
  same_bits ~msg:"each line's bits" by_rows lined

(* Divisions *)

(* The divisors device.h's [nx_divide] is checked at: 1, every power of two and
   its neighbours, every extent to 300, and drawn ones past 2^61 and anywhere,
   each with numerators at its multiples' edges, past 2^63, and drawn. *)
let divisions () =
  let st = Random.State.make [| 11 |] in
  let ds =
    [ 1 ]
    @ List.concat_map
        (fun k -> [ (1 lsl k) - 1; 1 lsl k; (1 lsl k) + 1 ])
        (List.init 61 succ)
    @ List.init 300 succ
    @ List.init 64 (fun _ ->
        (1 lsl 61) + 1 + Random.State.full_int st ((1 lsl 61) - 2))
    @ List.init 64 (fun _ -> 1 + Random.State.full_int st (max_int - 1))
  in
  let ns d =
    let d = Int64.of_int d in
    let q = Random.State.bits64 st in
    let edge = Int64.mul (Int64.unsigned_div q d) d in
    [
      0L;
      1L;
      -1L;
      Int64.min_int;
      Int64.pred Int64.min_int;
      edge;
      Int64.pred edge;
      Int64.add edge (Int64.pred d);
      Random.State.bits64 st;
      Int64.logor Int64.min_int (Random.State.bits64 st);
    ]
  in
  List.concat_map (fun d -> List.map (fun n -> (n, d)) (ns d)) ds

let buffer_of d (xs : int64 array) =
  let b = Bytes.create (8 * Array.length xs) in
  Array.iteri (fun i x -> Bytes.set_int64_le b (8 * i) x) xs;
  fst (device_of d b)

(* device.h's division by a divisor gives the unsigned quotient. *)
let test_divide () =
  let cases = Array.of_list (divisions ()) in
  let count = Array.length cases in
  let d = S.device () in
  let ns = buffer_of d (Array.map fst cases)
  and ds = buffer_of d (Array.map (fun (_, d) -> Int64.of_int d) cases)
  and ms = buffer_of d (Array.map (fun (_, d) -> F.divisor d) cases)
  and out = Rig.Buffer.create d (8 * count) in
  let threads = 256 in
  S.launch "divide"
    ~groups:((count + threads - 1) / threads)
    ~threads ~shared:0
    [ A out; A ns; A ds; A ms; W count ];
  let got = read_back out in
  Array.iteri
    (fun i (n, dv) ->
      equal
        ~msg:(Printf.sprintf "%Lu / %d" n dv)
        int64
        (Int64.unsigned_div n (Int64.of_int dv))
        (Bytes.get_int64_le got (8 * i)))
    cases

(* The values float32 arithmetic is checked at: the special ones, and normal
   ones of both signs whose products and sums fall below the least normal, two
   whose product rounds up to it, or pass the greatest finite. *)
let operands =
  Array.append specials
    (Array.map Int32.bits_of_float
       [|
         1.;
         -1.;
         0.5;
         3.;
         Float.ldexp 1. (-100);
         Float.ldexp (-1.) (-30);
         Float.ldexp 1. 100;
         Float.ldexp 1. (-63);
         Float.ldexp (1. -. Float.ldexp 1. (-24)) (-63);
       |])

let floats_of d (xs : int32 array) =
  let b = Bytes.create (4 * Array.length xs) in
  Array.iteri (fun i x -> Bytes.set_int32_le b (4 * i) x) xs;
  fst (device_of d b)

(* The model's float32 product, sum and fused multiply-add are the GPU's, bit
   for bit but for a NaN's payload, over every triple of [operands]. *)
let test_arith () =
  let k = Array.length operands in
  let count = k * k * k in
  let at i = Int32.float_of_bits operands.(i) in
  let triple i = (i / (k * k), i / k mod k, i mod k) in
  let pick f = Array.init count (fun i -> operands.(f (triple i))) in
  let d = S.device () in
  let a = floats_of d (pick (fun (a, _, _) -> a))
  and b = floats_of d (pick (fun (_, b, _) -> b))
  and c = floats_of d (pick (fun (_, _, c) -> c))
  and out = Rig.Buffer.create d (12 * count) in
  let threads = 256 in
  S.launch "arith"
    ~groups:((count + threads - 1) / threads)
    ~threads ~shared:0
    [ A out; A a; A b; A c; W count ];
  let got = read_back out in
  let bits x = Int32.bits_of_float x in
  let same x y = Int32.equal x y || (nan_bits x && nan_bits y) in
  for i = 0 to count - 1 do
    let ia, ib, ic = triple i in
    let x = at ia and y = at ib and z = at ic in
    List.iteri
      (fun j (name, model) ->
        let g = Bytes.get_int32_le got (4 * ((3 * i) + j)) in
        if not (same g (bits model)) then
          failf "%s of %h, %h, %h: %h, the model's %h" name x y z
            (Int32.float_of_bits g) model)
      [ ("product", mul x y); ("sum", add x y); ("fma", fma x y z) ]
  done

let test_moves () =
  List.iter test_case cases;
  equal ~msg:"transposes reached" (list string) all_moves
    (List.filter (Hashtbl.mem reached) all_moves);
  equal ~msg:"pairs reached" (list string) all_pairs
    (List.filter (Hashtbl.mem reached) all_pairs);
  equal ~msg:"a side's axes past the words" (list string) all_tailed
    (List.filter (Hashtbl.mem tailed) all_tailed)

let () =
  exit
    (run "nx_metal_fft_kernels"
       [
         group "moves"
           [
             test
               "each transpose, split and pre-pass writes the points fft.mli \
                says"
               test_moves;
             test "a split and a pre-pass compute their definitions"
               test_defined;
             test "an odd c2r's completion writes bin 0's conjugate" test_bin0;
             test "a step of 4096 points writes its model's bits" test_step;
             test "a column step writes the step's bits" test_column_step;
             test "a specialised step writes its model's bits" test_key_step;
             test "a long line's steps are FFTs alone" test_unrun_step;
             prop ~count:24 ~examples:delta_edges
               "the step's model is the DFT within its bound" delta_case
               law_step_dft;
             prop ~count:200 ~examples:key_delta_edges
               "a specialised step's model is the DFT within its bound"
               key_delta_case law_key_dft;
             test "an inverse c2c is the conjugated forward of its conjugate"
               test_conj_law;
             test "Nx_metal.c2c runs the calls it has kernels for" test_c2c;
             test "a wide side's line runs around the steps, in its bits"
               test_wide;
             test "the wide bound keeps a step's rows narrow" test_wide_bound;
             test "Nx_metal.c2c of 4096 x 4096 runs its plan" test_c2c2;
             test "Nx_metal.c2c of 4096 x 4096 runs its plan, a part a launch"
               (fun () -> S.parted ~launches:1 test_c2c2);
             test "a part between two passes through workspace keeps the bits"
               test_passes_parted;
             prop ~count:4 ~examples:dense_examples
               "dense lines and tones lie within the bound" dense_gen law_dense;
             test "a line below 2^115 overflows nowhere" test_overflow;
             test "a real or long line below its threshold overflows nowhere"
               test_real_overflow;
             test
               "a convolution or a pre-pass at its threshold overflows nowhere"
               test_convolution_overflow;
             key_caches;
             prop ~count:24 ~examples:layout_examples
               "a line's bits depend on its points alone" layout_gen law_layout;
             prop ~count:8 ~examples:layout_examples
               "a line's bits depend on its points alone, a part a launch"
               layout_gen (fun c ->
                 S.parted ~launches:1 (fun () -> law_layout ~parted:true c));
             test "a division by a divisor is the quotient" test_divide;
             test "the model's float32 arithmetic is the GPU's" test_arith;
           ];
       ])
