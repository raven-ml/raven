(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Quantized floats (FITS 4.0 §10.2), as the archives' files hold them: all
   written by cfitsio or code derived from it, whose bytes differ from the
   standard's text in the zero value of SUBTRACTIVE_DITHER_2 and in the
   dither index, which restarts at 10000 from (tile + ZDITHER0 - 2) mod
   10000 for 1-based tiles. *)

type ints = (int, Bigarray.int_elt, Bigarray.c_layout) Bigarray.Array1.t
type floats = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
type dither = No_dither | Subtractive_1 | Subtractive_2

let null_value = -2147483647
let zero_value = -2147483646
let n_reserved = 10
let n_random = 10000
let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Rounding to float32 through a one-element float32 array [c], which boxes
   nothing in a loop. *)
let round32 c x =
  Bigarray.Array1.unsafe_set c 0 x;
  Bigarray.Array1.unsafe_get c 0

(* Appendix I: 10000 uniform deviates from Park and Miller's generator,
   rounded to float32. *)
let randoms =
  lazy
    (let a = 16807. and m = 2147483647. in
     let seed = ref 1. in
     Array.init n_random (fun _ ->
         let temp = a *. !seed in
         seed := temp -. (m *. Float.of_int (int_of_float (temp /. m)));
         to_f32 (!seed /. m)))

(* The dither sequence of a tile: [start (tile + zdither0 - 1)] for a 0-based
   tile, then [next] after each pixel. The decoder scales the deviate in
   float32 and the encoder in float64, as cfitsio does. *)
type seq = {
  r : float array;
  mutable iseed : int;
  mutable next : int;
  f32 : bool;
}

let index s r =
  if s.f32 then int_of_float (to_f32 (r *. 500.)) else int_of_float (r *. 500.)

let start ~f32 row =
  let r = Lazy.force randoms in
  let iseed = (row - 1) mod n_random in
  let s = { r; iseed; next = 0; f32 } in
  s.next <- index s r.(iseed);
  s

let advance s =
  s.next <- s.next + 1;
  if s.next = n_random then begin
    s.iseed <- (if s.iseed + 1 = n_random then 0 else s.iseed + 1);
    s.next <- index s s.r.(s.iseed)
  end

(* [dequantize ~dither ~row ~scale ~zero ~blank i n out] writes the
   physical values of the [n] quantized [i] to [out]: [I × ZSCALE + ZZERO],
   or [((I − R) + 0.5) × ZSCALE + ZZERO] dithered, each operation rounded in
   float64, NaN for [blank] and 0 for the zero value of
   SUBTRACTIVE_DITHER_2. [row] is [tile + ZDITHER0] for a 0-based tile. *)
let dequantize ~dither ~row ~scale ~zero ~blank (i : ints) n (out : floats) =
  let is_blank v = match blank with Some b -> v = b | None -> false in
  match dither with
  | No_dither ->
      for k = 0 to n - 1 do
        let v = Bigarray.Array1.unsafe_get i k in
        Bigarray.Array1.unsafe_set out k
          (if is_blank v then Float.nan else (Float.of_int v *. scale) +. zero)
      done
  | Subtractive_1 | Subtractive_2 ->
      let s = start ~f32:true row in
      for k = 0 to n - 1 do
        let v = Bigarray.Array1.unsafe_get i k in
        let x =
          if is_blank v then Float.nan
          else if dither = Subtractive_2 && v = zero_value then 0.
          else ((Float.of_int v -. s.r.(s.next) +. 0.5) *. scale) +. zero
        in
        Bigarray.Array1.unsafe_set out k x;
        advance s
      done

(* Noise *)

(* [select a n k] is the [k]-th smallest of [a.(0)] to [a.(n - 1)], which
   it reorders (Hoare's selection). *)
let select (a : float array) n k =
  let lo = ref 0 and hi = ref (n - 1) in
  while !lo < !hi do
    let pivot = a.((!lo + !hi) / 2) in
    let i = ref !lo and j = ref !hi in
    while !i <= !j do
      while a.(!i) < pivot do
        incr i
      done;
      while a.(!j) > pivot do
        decr j
      done;
      if !i <= !j then begin
        let t = a.(!i) in
        a.(!i) <- a.(!j);
        a.(!j) <- t;
        incr i;
        decr j
      end
    done;
    if k <= !j then hi := !j else if k >= !i then lo := !i else lo := !hi
  done;
  a.(k)

(* The median of [a.(0)] to [a.(n - 1)], the lower one of an even count. *)
let lower_median (a : float array) n = select (Array.sub a 0 n) n ((n - 1) / 2)

let mid_median (a : float array) n =
  let c = Array.sub a 0 n in
  Array.sort Float.compare c;
  (c.((n - 1) / 2) +. c.(n / 2)) /. 2.

type stats = {
  ngood : int;
  minval : float;
  maxval : float;
  noise2 : float;
  noise3 : float;
  noise5 : float;
}

(* cfitsio's FnNoise5_float: the 2nd, 3rd and 5th order median absolute
   differences along each row of [ny] rows of [nx] pixels, NaN pixels
   skipped, then the median over the rows, in float32 arithmetic. Its 2nd
   order median runs over as many differences as the 3rd order's, stale ones
   included, as cfitsio's does. *)
let noise5 (data : floats) nx ny =
  let nx, ny = if nx < 9 then (nx * ny, 1) else (nx, ny) in
  let good v = not (Float.is_nan v) in
  let minv = ref Float.max_float and maxv = ref (-.Float.max_float) in
  let range v =
    if v < !minv then minv := v;
    if v > !maxv then maxv := v
  in
  if nx < 9 then begin
    let ngood = ref 0 in
    for k = 0 to nx - 1 do
      let v = Bigarray.Array1.get data k in
      if good v then (
        range v;
        incr ngood)
    done;
    {
      ngood = !ngood;
      minval = !minv;
      maxval = !maxv;
      noise2 = 0.;
      noise3 = 0.;
      noise5 = 0.;
    }
  end
  else begin
    let d2 = Array.make nx 0.
    and d3 = Array.make nx 0.
    and d5 = Array.make nx 0. in
    let r2 = Array.make ny 0.
    and r3 = Array.make ny 0.
    and r5 = Array.make ny 0. in
    let nrows = ref 0 and nrows2 = ref 0 and ngood = ref 0 in
    let cell = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 1 in
    let f = round32 cell in
    for jj = 0 to ny - 1 do
      let row = jj * nx in
      let ii = ref 0 in
      let next () =
        while !ii < nx && not (good (Bigarray.Array1.get data (row + !ii))) do
          incr ii
        done;
        if !ii = nx then None
        else begin
          let v = Bigarray.Array1.get data (row + !ii) in
          range v;
          Some v
        end
      in
      let v = Array.make 9 0. in
      let rec first k =
        if k = 8 then true
        else begin
          if k > 0 then incr ii;
          match next () with
          | None -> false
          | Some x ->
              v.(k) <- x;
              incr ngood;
              first (k + 1)
        end
      in
      if first 0 then begin
        let nvals = ref 0 and nvals2 = ref 0 in
        incr ii;
        let continue = ref true in
        while !continue && !ii < nx do
          match next () with
          | None -> continue := false
          | Some x ->
              v.(8) <- x;
              let v1 = v.(0)
              and v3 = v.(2)
              and v4 = v.(3)
              and v5 = v.(4)
              and v6 = v.(5)
              and v7 = v.(6)
              and v9 = v.(8) in
              if not (v5 = v6 && v6 = v7) then begin
                d2.(!nvals2) <- f (Float.abs (f (v5 -. v7)));
                incr nvals2
              end;
              if not (v3 = v4 && v4 = v5 && v5 = v6 && v6 = v7) then begin
                d3.(!nvals) <- f (Float.abs (f (f (f (2. *. v5) -. v3) -. v7)));
                d5.(!nvals) <-
                  f
                    (Float.abs
                       (f
                          (f
                             (f
                                (f (f (6. *. v5) -. f (4. *. v3))
                                -. f (4. *. v7))
                             +. v1)
                          +. v9)));
                incr nvals
              end
              else incr ngood;
              Array.blit v 1 v 0 8;
              incr ii
        done;
        ngood := !ngood + !nvals;
        if !nvals = 1 then begin
          if !nvals2 = 1 then (
            r2.(!nrows2) <- d2.(0);
            incr nrows2);
          r3.(!nrows) <- d3.(0);
          r5.(!nrows) <- d5.(0);
          incr nrows
        end
        else if !nvals > 1 then begin
          if !nvals2 > 1 then (
            r2.(!nrows2) <- lower_median d2 !nvals;
            incr nrows2);
          r3.(!nrows) <- lower_median d3 !nvals;
          r5.(!nrows) <- lower_median d5 !nvals;
          incr nrows
        end
      end
    done;
    let over r n =
      if n = 0 then 0. else if n = 1 then r.(0) else mid_median r n
    in
    {
      ngood = !ngood;
      minval = !minv;
      maxval = !maxv;
      noise2 = 1.0483579 *. over r2 !nrows2;
      noise3 = 0.6052697 *. over r3 !nrows;
      noise5 = 0.1772048 *. over r5 !nrows;
    }
  end

let nint x = if x >= 0. then int_of_float (x +. 0.5) else int_of_float (x -. 0.5)

(* [quantize ~dither ~row ~q data nx ny out] is cfitsio's
   fits_quantize_float: the integers of the tile [data] in steps of its
   noise divided by [q], NaN as the null value, written to [out], with
   ZSCALE and ZZERO; [None] where the tile has no noise or a range past
   int32, which then stays lossless. *)
let quantize ~dither ~row ~q (data : floats) nx ny (out : ints) =
  let n = nx * ny in
  if n <= 1 then None
  else
    let s = noise5 data nx ny in
    let minval, maxval, stdev =
      if s.ngood = 0 then (0., 1., 1.)
      else
        let sd = s.noise3 in
        let sd = if s.noise2 <> 0. && s.noise2 < sd then s.noise2 else sd in
        let sd = if s.noise5 <> 0. && s.noise5 < sd then s.noise5 else sd in
        (s.minval, s.maxval, sd)
    in
    let delta = stdev /. q in
    if delta = 0. then None
    else if
      (maxval -. minval) /. delta
      > (2. *. 2147483647.) -. Float.of_int n_reserved
    then None
    else begin
      let seq = start ~f32:false row in
      let zeropt =
        if s.ngood = n then
          if dither = Subtractive_2 then
            minval -. (delta *. Float.of_int (null_value + n_reserved))
          else if
            (maxval -. minval) /. delta < 2147483647. -. Float.of_int n_reserved
          then
            Float.of_int
              (Int64.to_int (Int64.of_float ((minval /. delta) +. 0.5)))
            *. delta
          else (minval +. maxval) /. 2.
        else minval -. (delta *. Float.of_int (null_value + n_reserved))
      in
      for k = 0 to n - 1 do
        let x = Bigarray.Array1.unsafe_get data k in
        let v =
          if Float.is_nan x then null_value
          else if dither = Subtractive_2 && x = 0. then zero_value
          else if dither = No_dither then nint ((x -. zeropt) /. delta)
          else nint (((x -. zeropt) /. delta) +. seq.r.(seq.next) -. 0.5)
        in
        Bigarray.Array1.unsafe_set out k v;
        if dither <> No_dither then advance seq
      done;
      Some (delta, zeropt)
    end
