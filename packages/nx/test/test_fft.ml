(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.mli's Fourier transforms, through Nx: each against the sum it defines,
   computed in OCaml at float64, within the bound nx.mli states over the host
   library's, nx.cpu's, and the identities between them. Then Prim.Fft, the
   operation they apply. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let pi = Float.pi
let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let z re im = { Complex.re; im }

let parts (a : Complex.t array) =
  Array.map (fun (c : Complex.t) -> (c.re, c.im)) a

let exact_parts = array (pair float_exact float_exact)

(* Bounds *)

(* nx.cpu's constant for a transform of [n] points (nx_cpu.mli, Fourier
   transforms): [8 Σ p^(3/2) + 8] over [n]'s prime factors. No length here has a
   prime factor past 250. *)
let e n =
  let rec go n p acc =
    if n <= 1 then acc
    else if n mod p = 0 then go (n / p) p (acc +. Float.pow (Float.of_int p) 1.5)
    else go n (p + 1) acc
  in
  (8. *. go n 2 0.) +. 8.

let norm2 (a : Complex.t array) =
  Float.sqrt (Array.fold_left (fun s v -> s +. Complex.norm2 v) 0. a)

let sum_abs (a : Complex.t array) =
  Array.fold_left (fun s v -> s +. Complex.norm v) 0. a

(* The error of [dft] below, in float64's [u] times [Σ |x|]: each twiddle's
   parts within [(6π + 1) u] (three roundings of its phase, libm's ulp), so [√2
   (6π + 1) u] in modulus, the complex product [√5 u], and [n - 1] additions. *)
let reference n =
  (Float.sqrt 2. *. ((6. *. pi) +. 1.)) +. Float.sqrt 5. +. Float.of_int (n - 1)

(* nx.mli's allowance for a transform of [points] points with constant [eps],
   scaled by [c]: [c ε u √N ‖x‖₂], plus the reference's error over lines of [n]
   points. At complex128 the reference errs by about as much as the bound, so
   those draws check what nx adds (axes, lengths, scaling, bins); the complex64
   draws, where the float64 reference is 2^-29 of the bound, check the bound. *)
let allowance ~u ~c ~eps ~points ~n ~norm ~abs =
  c
  *. ((eps *. u *. Float.sqrt points *. norm) +. (reference n *. 0x1p-53 *. abs))

(* [‖got - want‖₂], each element's error a [Nx_support.complex_gap], so that a
   NaN or a misplaced infinity is infinitely far. *)
let distance (want : Complex.t array) (got : Complex.t array) =
  let s = ref 0. in
  Array.iteri
    (fun k w ->
      let d = Nx_support.complex_gap got.(k) w in
      s := !s +. (d *. d))
    want;
  Float.sqrt !s

(* The error's 2-norm within [tol], plus [3u ‖want‖₂] for the scaling: the
   bounds nx.mli and nx_cpu.mli state are of [‖ŷ - y‖₂]. *)
let within ~msg ~u tol (want : Complex.t array) (got : Complex.t array) =
  equal ~msg:(msg ^ ", length") int (Array.length want) (Array.length got);
  at_most
    ~msg:(msg ^ ", ‖ŷ - y‖₂")
    float_exact
    ~than:(tol +. (3. *. u *. norm2 want))
    (distance want got)

let reals a = Array.map (fun re -> z re 0.) a

(* [Σ_j x[j] e^(sign 2πi jk/n)] for each [k]. *)
let dft ~sign (x : Complex.t array) =
  let n = Array.length x in
  Array.init n (fun k ->
      let acc = ref Complex.zero in
      Array.iteri
        (fun j v ->
          let t =
            sign *. 2. *. pi *. Float.of_int (j * k mod n) /. Float.of_int n
          in
          acc := Complex.add !acc (Complex.mul v { re = cos t; im = sin t }))
        x;
      !acc)

(* The factor of a transform of [n] points, forward or not, under [norm]. *)
let factor norm ~forward n =
  let f = Float.of_int (Stdlib.max 1 n) in
  match (norm, forward) with
  | Nx.Backward, true | Nx.Forward, false -> 1.
  | Nx.Backward, false | Nx.Forward, true -> 1. /. f
  | Nx.Ortho, _ -> 1. /. Float.sqrt f

let norms = [ Nx.Backward; Nx.Forward; Nx.Ortho ]

let norm_name = function
  | Nx.Backward -> "Backward"
  | Forward -> "Forward"
  | Ortho -> "Ortho"

let scale c a = Array.map (fun v -> Complex.mul (z c 0.) v) a

(* Signals *)

(* A complex dtype, the real one of its precision, and its unit roundoff. *)
type precision =
  | Precision : (Complex.t, 's) D.t * (float, 'r) D.t * float -> precision

let precisions =
  [
    Precision (D.Complex64, D.Float32, 0x1p-24);
    Precision (D.Complex128, D.Float64, 0x1p-53);
  ]

let pp_precision ppf (Precision (c, _, _)) = D.pp ppf c

(* Parts of magnitudes from [2^-20]: the bounds hold barring underflow. *)
let part =
  Gen.map
    (fun (neg, m) -> if neg then -.m else m)
    (Gen.pair Gen.bool (Gen.float_range 0x1p-20 1.))

(* [n] parts in [-1, 1], [n] from 0 to 9, at a drawn precision. *)
let signal =
  let open Gen in
  let* p = of_list ~pp:pp_precision precisions in
  let* n = int_range 0 9 in
  let+ re = array ~size:(constant n) part
  and+ im = array ~size:(constant n) part in
  (p, Array.map2 z re im)

(* Signals of no point and of one at each precision. *)
let short =
  List.concat_map (fun p -> [ (p, [||]); (p, [| z 0.5 (-0.25) |]) ]) precisions

(* A value's elements at float64. *)
let read y = Nx.to_array (Nx.cast Nx.complex128 y)
let read_real y = Nx.to_array (Nx.cast Nx.float64 y)

(* Demands that a law meet each length the generator draws. *)
let cover_lengths n =
  cover "no point" (n = 0);
  cover "one point" (n = 1);
  cover "an odd length" (n > 1 && n mod 2 = 1);
  cover "an even length" (n > 0 && n mod 2 = 0)

let cover_precision (Precision (c, _, _)) =
  List.iter
    (fun (Precision (d, _, _)) -> cover (D.name d) (D.name c = D.name d))
    precisions

(* fft and ifft *)

(* A round trip: each transform within its bound, the second applying the
   first's error and scaling through an operator of norm [ci √N]; with [cf ci N
   = 1] that is [(2ε + 6) u ‖x‖₂]. *)
let trip ~u ~eps norm = ((2. *. eps) +. 6.) *. u *. norm

let law_fft ((Precision (dt, _, u) as p), xs0) =
  let n = Array.length xs0 in
  cover_lengths n;
  cover_precision p;
  let x = Nx.create dt [| n |] xs0 in
  let xs = read x in
  let nm = norm2 xs and ab = sum_abs xs in
  let tol c =
    allowance ~u ~c ~eps:(e n) ~points:(Float.of_int n) ~n ~norm:nm ~abs:ab
  in
  List.iter
    (fun norm ->
      let msg = norm_name norm in
      let cf = factor norm ~forward:true n
      and ci = factor norm ~forward:false n in
      within ~msg:(msg ^ ", fft") ~u (tol cf)
        (scale cf (dft ~sign:(-1.) xs))
        (read (Nx.fft ~norm x));
      within ~msg:(msg ^ ", ifft") ~u (tol ci)
        (scale ci (dft ~sign:1. xs))
        (read (Nx.ifft ~norm x));
      within ~msg:(msg ^ ", ifft (fft x)") ~u
        (trip ~u ~eps:(e n) nm)
        xs
        (read (Nx.ifft ~norm (Nx.fft ~norm x))))
    norms

(* Along two axes in either order, the transform along each in turn, scaled over
   all their [m n] points. The reference runs the two transforms at complex128
   on [x]'s elements, within its own bound. *)
let law_axes (Precision (dt, _, u), (m, n)) =
  let x =
    Nx.create dt [| m; n |]
      (Array.init (m * n) (fun i ->
           z (sin (Float.of_int (i * 7))) (cos (Float.of_int (i * 3)))))
  in
  let x64 = Nx.cast Nx.complex128 x in
  let xs = read x in
  let eps = e m +. e n and points = Float.of_int (m * n) in
  let nm = norm2 xs in
  let turns = read (Nx.fft ~axes:[ 0 ] (Nx.fft ~axes:[ 1 ] x64)) in
  List.iter
    (fun norm ->
      let c = factor norm ~forward:true (m * n) in
      let tol =
        allowance ~u ~c ~eps ~points ~n:1 ~norm:nm ~abs:0.
        +. allowance ~u:0x1p-53 ~c ~eps ~points ~n:1 ~norm:nm ~abs:0.
      in
      let msg = norm_name norm in
      within ~msg:(msg ^ ", axes [0; 1]") ~u tol (scale c turns)
        (read (Nx.fft ~norm ~axes:[ 0; 1 ] x));
      within ~msg:(msg ^ ", axes [1; 0]") ~u tol (scale c turns)
        (read (Nx.fft ~norm ~axes:[ 1; 0 ] x)))
    norms;
  (* The real axis last, or first: rfft's bins are fft's along it. *)
  let r =
    Nx.create Nx.float64 [| m; n |]
      (Array.map (fun (c : Complex.t) -> c.re) (read x64))
  in
  let both = read (Nx.fft ~axes:[ 0; 1 ] (Nx.cast Nx.complex128 r)) in
  let first = read (Nx.rfft Nx.complex128 ~axes:[ 1; 0 ] r) in
  let bins = (m / 2) + 1 in
  let tol =
    2.
    *. allowance ~u:0x1p-53 ~c:1. ~eps ~points ~n:1
         ~norm:(norm2 (reals (Nx.to_array r)))
         ~abs:0.
  in
  within ~msg:"rfft, the real axis first" ~u:0x1p-53 tol
    (Array.sub both 0 (bins * n))
    first

let test_sizes () =
  let x = Nx.create Nx.complex128 [| 3 |] [| z 1. 0.; z 2. 0.; z 3. 0. |] in
  let padded =
    Nx.create Nx.complex128 [| 5 |]
      [| z 1. 0.; z 2. 0.; z 3. 0.; Complex.zero; Complex.zero |]
  in
  equal ~msg:"extended" exact_parts
    (parts (Nx.to_array (Nx.fft padded)))
    (parts (Nx.to_array (Nx.fft ~s:[ 5 ] x)));
  equal ~msg:"cut" exact_parts
    (parts (Nx.to_array (Nx.fft (Nx.slice [ Nx.R (0, 2) ] x))))
    (parts (Nx.to_array (Nx.fft ~s:[ 2 ] x)));
  equal ~msg:"no point" int 0 (Nx.numel (Nx.fft ~s:[ 0 ] x))

(* The scaling multiplies each part by the real factor: an infinite part keeps
   its zero imaginary part, where a complex product by [1/2 + 0i] gives NaN. *)
let test_scaled_parts () =
  let inf =
    Nx.create Nx.complex128 [| 2 |] [| z Float.infinity 0.; Complex.zero |]
  in
  equal exact_parts
    [| (Float.infinity, 0.); (Float.infinity, 0.) |]
    (parts (Nx.to_array (Nx.fft ~norm:Forward inf)))

(* rfft and irfft *)

(* rfft is fft's first [n/2 + 1] bins, one bin of +0 for no point, and irfft of
   them is the signal, under each norm. A c2r's operand norm is √2 that of its
   bins (nx_cpu.mli). *)
let law_real ((Precision (cdt, rdt, u) as p), xs0) =
  let n = Array.length xs0 in
  cover_lengths n;
  cover_precision p;
  let x = Nx.create rdt [| n |] (Array.map (fun (c : Complex.t) -> c.re) xs0) in
  let rs = read_real x in
  let xs = reals rs in
  let nm = norm2 xs and ab = sum_abs xs in
  let full = dft ~sign:(-1.) xs in
  List.iter
    (fun norm ->
      let msg = norm_name norm in
      let c = factor norm ~forward:true n in
      let bins = Nx.rfft cdt ~norm x in
      if n = 0 then
        equal ~msg:(msg ^ ", no point") exact_parts
          [| (0., 0.) |]
          (parts (read bins))
      else begin
        let tol =
          allowance ~u ~c ~eps:(e n) ~points:(Float.of_int n) ~n ~norm:nm
            ~abs:ab
        in
        within ~msg:(msg ^ ", rfft") ~u tol
          (scale c (Array.sub full 0 ((n / 2) + 1)))
          (read bins);
        within ~msg:(msg ^ ", irfft (rfft x)") ~u
          (Float.sqrt 2. *. trip ~u ~eps:(e n) nm)
          xs
          (reals (read_real (Nx.irfft rdt ~norm ~s:[ n ] bins)))
      end)
    norms

(* irfft's linear map, as nx.mli states it: the bins cut or extended to [n/2 +
   1], bin [k] above [n/2] the conjugate of bin [n - k], the imaginary parts of
   bin 0 and, for an even [n], bin [n/2] unread. *)
let hermitian (zs : Complex.t array) n =
  let h = (n / 2) + 1 in
  let bin k =
    let b = if k < Array.length zs then zs.(k) else Complex.zero in
    if k = 0 || 2 * k = n then { b with im = 0. } else b
  in
  Array.init n (fun k -> if k < h then bin k else Complex.conj (bin (n - k)))

let law_irfft (m, n) =
  cover "bins cut" (m > (n / 2) + 1);
  cover "bins extended" (m < (n / 2) + 1);
  cover "an even n" (n mod 2 = 0);
  cover "an odd n" (n mod 2 = 1);
  let zs =
    Array.init m (fun k ->
        z (cos (Float.of_int ((3 * k) + 1))) (sin (Float.of_int ((5 * k) + 2))))
  in
  let t = Nx.create Nx.complex128 [| m |] zs in
  let check ~msg n y =
    let full = hermitian zs n in
    let c = factor Nx.Backward ~forward:false n in
    let tol =
      allowance ~u:0x1p-53 ~c ~eps:(e n) ~points:(Float.of_int n) ~n
        ~norm:(Float.sqrt 2. *. norm2 full)
        ~abs:(sum_abs full)
    in
    within ~msg ~u:0x1p-53 tol
      (reals (Array.map (fun (v : Complex.t) -> c *. v.re) (dft ~sign:1. full)))
      (reals (Nx.to_array y))
  in
  check ~msg:"n points" n (Nx.irfft Nx.float64 ~s:[ n ] t);
  let n' = 2 * (m - 1) in
  if n' > 0 then check ~msg:"2 (m - 1) points" n' (Nx.irfft Nx.float64 t)

(* irfft reads no imaginary part of bin 0 or, at an even n, of bin n/2: NaN and
   infinities there give the bits +0 gives. *)
let test_unread_parts () =
  List.iter
    (fun n ->
      let h = (n / 2) + 1 in
      let bins im =
        Nx.create Nx.complex128 [| h |]
          (Array.init h (fun k ->
               z
                 (Float.of_int (k + 1))
                 (if k = 0 || 2 * k = n then im k else 0.5)))
      in
      equal
        ~msg:(Printf.sprintf "%d points" n)
        (array float_exact)
        (Nx.to_array (Nx.irfft Nx.float64 ~s:[ n ] (bins (fun _ -> 0.))))
        (Nx.to_array
           (Nx.irfft Nx.float64 ~s:[ n ]
              (bins (fun k -> if k = 0 then Float.nan else Float.infinity)))))
    [ 6; 7; 8 ]

(* A float narrower than float32 transforms at float32: rfft of it has the bits
   of rfft of its float32 cast, and irfft into it is the float32 result rounded
   once. The signal is one where a transform at float64 rounded once gives other
   bits: for rfft at every format, for irfft at float16 and bfloat16, while a
   float8 result's three bits of fraction store the same from either precision,
   so there it holds the rounding alone. *)
let test_narrow () =
  let x =
    Nx.create Nx.float32 [| 256 |]
      (Array.init 256 (fun i -> 3. *. sin (Float.of_int (i * i))))
  in
  let narrow (type s) (dt : (float, s) Nx.dtype) =
    let h = Nx.cast dt x in
    let wide = Nx.cast Nx.float32 h in
    let bins z = Nx.to_array (Nx.bitcast Nx.int32 (Nx.bitcast Nx.float32 z)) in
    equal
      ~msg:("rfft of " ^ Nx.Dtype.name dt)
      (array int32)
      (bins (Nx.rfft Nx.complex64 wide))
      (bins (Nx.rfft Nx.complex64 h));
    let z = Nx.rfft Nx.complex64 wide in
    let points y = Nx.to_array (Nx.bitcast Nx.int64 (Nx.cast Nx.float64 y)) in
    equal
      ~msg:("irfft into " ^ Nx.Dtype.name dt)
      (array int64)
      (points (Nx.cast dt (Nx.irfft Nx.float32 z)))
      (points (Nx.irfft dt z))
  in
  narrow Nx.float16;
  narrow Nx.bfloat16;
  narrow Nx.float8_e4m3fn;
  narrow Nx.float8_e5m2

(* rfft and irfft compute at the wider of the operand's and the result's
   precision and round once into the result's dtype. *)
let test_wider () =
  let x =
    Nx.create Nx.float64 [| 257 |]
      (Array.init 257 (fun i -> sin (Float.of_int (i * i))))
  in
  let x32 = Nx.cast Nx.float32 x in
  let bits32 y = Nx.to_array (Nx.bitcast Nx.int32 (Nx.bitcast Nx.float32 y)) in
  let bits64 y = Nx.to_array (Nx.bitcast Nx.int64 (Nx.bitcast Nx.float64 y)) in
  equal ~msg:"rfft complex64 of float64" (array int32)
    (bits32 (Nx.cast Nx.complex64 (Nx.rfft Nx.complex128 x)))
    (bits32 (Nx.rfft Nx.complex64 x));
  equal ~msg:"rfft complex128 of float32" (array int64)
    (bits64 (Nx.rfft Nx.complex128 (Nx.cast Nx.float64 x32)))
    (bits64 (Nx.rfft Nx.complex128 x32));
  let zs = Nx.rfft Nx.complex128 x in
  equal ~msg:"irfft float32 of complex128" (array int32)
    (Nx.to_array
       (Nx.bitcast Nx.int32 (Nx.cast Nx.float32 (Nx.irfft Nx.float64 zs))))
    (Nx.to_array (Nx.bitcast Nx.int32 (Nx.irfft Nx.float32 zs)));
  let z64 = Nx.cast Nx.complex64 zs in
  equal ~msg:"irfft float64 of complex64" (array int64)
    (Nx.to_array
       (Nx.bitcast Nx.int64 (Nx.irfft Nx.float64 (Nx.cast Nx.complex128 z64))))
    (Nx.to_array (Nx.bitcast Nx.int64 (Nx.irfft Nx.float64 z64)))

(* hfft and ihfft *)

let bins_and_length =
  let open Gen in
  let* m = frequency [ (1, constant 1); (2, int_range 2 6) ] and+ odd = bool in
  let+ re = array ~size:(constant m) part
  and+ im = array ~size:(constant m) part in
  (Array.map2 z re im, if odd then (2 * m) - 1 else 2 * (m - 1))

(* One bin read as no point and as one point. *)
let one_bin = [ ([| z 0.5 0.25 |], 0); ([| z 0.5 0.25 |], 1) ]

(* hfft is irfft of the conjugate scaled as a forward transform, so its bound is
   irfft's; ihfft is rfft's conjugate scaled as an inverse. *)
let law_hermitian (zs, n) =
  cover_lengths n;
  let m = Array.length zs in
  cover "one bin" (m = 1);
  let t = Nx.create Nx.complex128 [| m |] zs in
  let y = Array.init n (fun j -> zs.(j mod m).re -. zs.(j mod m).im) in
  let ty = Nx.create Nx.float64 [| n |] y in
  let u = 0x1p-53 in
  let full = hermitian zs n in
  List.iter
    (fun norm ->
      let msg = norm_name norm in
      let c = factor norm ~forward:true n in
      let want =
        reals
          (Array.map (fun (v : Complex.t) -> c *. v.re) (dft ~sign:(-1.) full))
      in
      let tol =
        allowance ~u ~c ~eps:(e n) ~points:(Float.of_int n) ~n
          ~norm:(Float.sqrt 2. *. norm2 full)
          ~abs:(sum_abs full)
      in
      within ~msg:(msg ^ ", hfft") ~u tol want
        (reals (Nx.to_array (Nx.hfft Nx.float64 ~norm ~s:[ n ] t)));
      if n mod 2 = 0 then
        within
          ~msg:(msg ^ ", hfft's default points")
          ~u tol want
          (reals (Nx.to_array (Nx.hfft Nx.float64 ~norm t)));
      if n > 0 then begin
        let ci = factor norm ~forward:false n in
        let ys = reals y in
        let inverse = scale ci (Array.sub (dft ~sign:1. ys) 0 ((n / 2) + 1)) in
        let h = Nx.ihfft Nx.complex128 ~norm ty in
        within ~msg:(msg ^ ", ihfft") ~u
          (allowance ~u ~c:ci ~eps:(e n) ~points:(Float.of_int n) ~n
             ~norm:(norm2 ys) ~abs:(sum_abs ys))
          inverse (Nx.to_array h);
        within ~msg:(msg ^ ", hfft (ihfft x)") ~u
          (Float.sqrt 2. *. trip ~u ~eps:(e n) (norm2 ys))
          ys
          (reals (Nx.to_array (Nx.hfft Nx.float64 ~norm ~s:[ n ] h)))
      end)
    norms

(* Over two axes, in either order: the half of a Hermitian signal, the inverse
   sums of a real spectrum, transforms back to that spectrum, and ihfft of the
   spectrum is that half. *)
let test_hermitian_axes () =
  let rows = 3 and cols = 4 in
  let spectrum =
    Array.init (rows * cols) (fun j -> Float.of_int (j * 7 mod 5) -. 2.)
  in
  let along_rows y =
    Array.concat
      (List.init rows (fun r -> dft ~sign:1. (Array.sub y (r * cols) cols)))
  in
  let along_cols y =
    let col c = Array.init rows (fun r -> y.((r * cols) + c)) in
    let t = Array.init cols (fun c -> dft ~sign:1. (col c)) in
    Array.init (rows * cols) (fun j -> t.(j mod cols).(j / cols))
  in
  let n = Float.of_int (rows * cols) in
  let h = scale (1. /. n) (along_cols (along_rows (reals spectrum))) in
  let ty = Nx.create Nx.float64 [| rows; cols |] spectrum in
  let th = Nx.create Nx.complex128 [| rows; cols |] h in
  let half axis =
    match axis with
    | 1 -> Nx.slice [ A; R (0, (cols / 2) + 1) ] th
    | _ -> Nx.slice [ R (0, (rows / 2) + 1); A ] th
  in
  let u = 0x1p-53 and eps = e rows +. e cols in
  let norm = norm2 (reals spectrum) in
  (* Each side within the two axes' bound over the twelve points, and the
     reference's own over the rows and columns it sums. *)
  let tol =
    2.
    *. allowance ~u ~c:1. ~eps ~points:n ~n:(rows + cols) ~norm
         ~abs:(sum_abs (reals spectrum))
  in
  List.iter
    (fun (axes, s) ->
      let msg =
        Printf.sprintf "axes [%s]"
          (String.concat "; " (List.map string_of_int axes))
      in
      let last = List.nth axes 1 in
      within ~msg:(msg ^ ", hfft") ~u
        (Float.sqrt 2. *. tol)
        (reals spectrum)
        (reals (Nx.to_array (Nx.hfft Nx.float64 ~axes ~s (half last))));
      within ~msg:(msg ^ ", ihfft") ~u (tol /. n)
        (Nx.to_array (half last))
        (Nx.to_array (Nx.ihfft Nx.complex128 ~axes ty)))
    [ ([ 0; 1 ], [ rows; cols ]); ([ 1; 0 ], [ cols; rows ]) ]

(* dct, dst *)

let types = [ (Nx.Type_1, 1); (Nx.Type_2, 2); (Nx.Type_3, 3); (Nx.Type_4, 4) ]

(* The cosine and sine transforms' sums, at float64, as nx.mli states them. *)
let reference_trig family kind (x : float array) =
  let n = Array.length x in
  let f = Float.of_int n in
  let sum lo hi g =
    let s = ref 0. in
    for j = lo to hi do
      s := !s +. g j
    done;
    !s
  in
  let sign k = if k mod 2 = 0 then 1. else -1. in
  Array.init n (fun k ->
      let k' = Float.of_int k in
      let j' = Float.of_int in
      match (family, kind) with
      | `C, 1 ->
          x.(0)
          +. (sign k *. x.(n - 1))
          +. 2.
             *. sum 1 (n - 2) (fun j ->
                 x.(j) *. cos (pi *. j' j *. k' /. (f -. 1.)))
      | `C, 2 ->
          2.
          *. sum 0 (n - 1) (fun j ->
              x.(j) *. cos (pi *. k' *. ((2. *. j' j) +. 1.) /. (2. *. f)))
      | `C, 3 ->
          x.(0)
          +. 2.
             *. sum 1 (n - 1) (fun j ->
                 x.(j) *. cos (pi *. j' j *. ((2. *. k') +. 1.) /. (2. *. f)))
      | `C, _ ->
          2.
          *. sum 0 (n - 1) (fun j ->
              x.(j)
              *. cos
                   (pi
                   *. ((2. *. j' j) +. 1.)
                   *. ((2. *. k') +. 1.)
                   /. (4. *. f)))
      | `S, 1 ->
          2.
          *. sum 0 (n - 1) (fun j ->
              x.(j) *. sin (pi *. (j' j +. 1.) *. (k' +. 1.) /. (f +. 1.)))
      | `S, 2 ->
          2.
          *. sum 0 (n - 1) (fun j ->
              x.(j)
              *. sin (pi *. (k' +. 1.) *. ((2. *. j' j) +. 1.) /. (2. *. f)))
      | `S, 3 ->
          (sign k *. x.(n - 1))
          +. 2.
             *. sum 0 (n - 2) (fun j ->
                 x.(j)
                 *. sin (pi *. (j' j +. 1.) *. ((2. *. k') +. 1.) /. (2. *. f)))
      | `S, _ ->
          2.
          *. sum 0 (n - 1) (fun j ->
              x.(j)
              *. sin
                   (pi
                   *. ((2. *. j' j) +. 1.)
                   *. ((2. *. k') +. 1.)
                   /. (4. *. f))))

(* nx.mli's [L], [a] and [b] of a type of [n] points at unit roundoff [u], and
   its logical length. Type II is [y_k = 2 Re (Z_k e^(-iθ_k))], [Z] the
   transform of [n] points, [θ_k = k fl(π/2n)] at float64, rounded once to
   float32 there: [θ < π/2], so [|Δθ| <= (π/2) (3u_64 + u_32)] at float32 and
   [(π/2) 3u_64] at float64. [|e^(iθ̂) - e^(iθ)| <= |Δθ|], and cos and sin within
   2 ulps, [2.5u] each on [[0, 1]], move [(cos, sin)] by [√2 2.5u]. The two
   products and the sum round within [u (|Re Z| c + |Im Z| s) + u |y_k| / 2 <=
   2u |Z_k|]. With [‖Z‖₂ = √n ‖x‖₂] and [a = 2], [b <= Δθ/u + 3.54 + 2]: 7.1 at
   float32 and 10.3 at float64, stated as 8 and 11. Type IV is type II over [2n]
   points. *)
let shape family kind n ~u =
  let b2 = if u > 0x1p-30 then 8. else 11. in
  let logical =
    match (family, kind) with
    | `C, 1 -> 2 * (n - 1)
    | `S, 1 -> 2 * (n + 1)
    | _ -> 2 * n
  in
  match kind with
  | 1 -> (logical, Float.sqrt 2., 0., logical)
  | 2 -> (n, 2., b2, logical)
  | 3 -> (4 * n, Float.sqrt 2., 0., logical)
  | _ -> (2 * n, 2., b2, logical)

(* The inverse type: III for II, II for III. *)
let inverse_kind = function 2 -> 3 | 3 -> 2 | k -> k

type real_precision = Real : (float, 's) D.t * float -> real_precision

let trig_signal =
  let open Gen in
  let* p =
    of_list
      ~pp:(fun ppf (Real (dt, _)) -> D.pp ppf dt)
      [ Real (D.Float32, 0x1p-24); Real (D.Float64, 0x1p-53) ]
  in
  let* n = int_range 0 9 in
  let+ x = array ~size:(constant n) part in
  (p, x)

let trig_short =
  List.concat_map
    (fun p -> [ (p, [||]); (p, [| 0.75 |]) ])
    [ Real (D.Float32, 0x1p-24); Real (D.Float64, 0x1p-53) ]

(* An axis of no element gives a copy of the shape. *)
let empty_trig (s, axes) =
  List.iter
    (fun (type_, _) ->
      List.iter
        (fun y -> equal (array int) s (Nx.shape y))
        [
          Nx.dct ~type_ ~axes (Nx.zeros Nx.float32 s);
          Nx.idct ~type_ ~axes (Nx.zeros Nx.float32 s);
          Nx.dst ~type_ ~axes (Nx.zeros Nx.float32 s);
          Nx.idst ~type_ ~axes (Nx.zeros Nx.float32 s);
        ])
    types

(* Each type is its sum within nx.mli's [a (ε + b) u √L ‖x‖₂] and the
   reference's [(3π + 3 + n) 2^-53 · 2 Σ|x|]; Forward divides by the logical
   length; idct and idst undo it under each norm, within the two bounds through
   the inverse's operator norm [a' √L'] over the logical length, twice that
   under Ortho, whose end weights [√2] and [1/√2] scale each side by at most
   [√2]; Ortho keeps the energy. *)
let law_trig (Real (dt, u), xs0) =
  let n = Array.length xs0 in
  cover_lengths n;
  if n = 0 then empty_trig ([| 0 |], [ 0 ])
  else
    let t = Nx.create dt [| n |] xs0 in
    let x = read_real t in
    let nx = norm2 (reals x) in
    let ref_err =
      ((3. *. pi) +. 3. +. Float.of_int n) *. 0x1p-53 *. 2. *. sum_abs (reals x)
    in
    List.iter
      (fun family ->
        List.iter
          (fun (type_, kind) ->
            if not (family = `C && kind = 1 && n = 1) then begin
              let forward, inverse =
                match family with
                | `C -> (Nx.dct, Nx.idct)
                | `S -> (Nx.dst, Nx.idst)
              in
              let msg =
                Printf.sprintf "%s type %d"
                  (match family with `C -> "dct" | `S -> "dst")
                  kind
              in
              let l, a, b, logical = shape family kind n ~u in
              let bound =
                (a *. (e l +. b) *. u *. Float.sqrt (Float.of_int l) *. nx)
                +. ref_err
              in
              let want = reference_trig family kind x in
              within ~msg ~u bound (reals want)
                (reals (read_real (forward ~type_ t)));
              let c = 1. /. Float.of_int logical in
              within ~msg:(msg ^ ", Forward") ~u (c *. bound)
                (reals (Array.map (fun v -> c *. v) want))
                (reals (read_real (forward ~type_ ~norm:Forward t)));
              let l', a', b', _ = shape family (inverse_kind kind) n ~u in
              let through =
                a' *. Float.sqrt (Float.of_int l') /. Float.of_int logical
              in
              let inverse_bound =
                a'
                *. (e l' +. b')
                *. u
                *. Float.sqrt (Float.of_int l')
                *. (a *. Float.sqrt (Float.of_int l) *. nx)
                /. Float.of_int logical
              in
              List.iter
                (fun norm ->
                  let weights = match norm with Nx.Ortho -> 2. | _ -> 1. in
                  within
                    ~msg:(msg ^ ", inverted under " ^ norm_name norm)
                    ~u
                    (weights
                     *. ((through *. (bound -. ref_err)) +. inverse_bound)
                    +. (6. *. u *. nx))
                    (reals x)
                    (reals
                       (read_real
                          (inverse ~type_ ~norm (forward ~type_ ~norm t)))))
                norms;
              let energy a = Array.fold_left (fun s v -> s +. (v *. v)) 0. a in
              let o = read_real (forward ~type_ ~norm:Ortho t) in
              (* Ortho's result within [√2 bound / √logical] of an orthonormal
                 transform's, whose norm is [‖x‖₂]. *)
              let d =
                (2. *. bound /. Float.sqrt (Float.of_int logical))
                +. (6. *. u *. nx)
              in
              at_most ~msg:(msg ^ ", Ortho's energy") float_exact
                ~than:((2. *. d *. nx) +. (d *. d))
                (Nx_support.gap (energy o) (energy x))
            end)
          types)
      [ `C; `S ]

(* Along two axes the transform along each in turn. *)
let law_trig_axes (m, n) =
  let x =
    Nx.init Nx.float64 [| m; n |] (fun i ->
        sin (Float.of_int ((i.(0) * 5) + (i.(1) * 3))))
  in
  let u = 0x1p-53 in
  let nx = norm2 (reals (Nx.to_array x)) in
  List.iter
    (fun (type_, kind) ->
      if not (kind = 1 && (m = 1 || n = 1)) then begin
        let l1, a1, b1, _ = shape `C kind n ~u
        and l0, a0, b0, _ = shape `C kind m ~u in
        let ops = a1 *. a0 *. Float.sqrt (Float.of_int (l1 * l0)) in
        let tol = 2. *. ops *. (e l1 +. b1 +. e l0 +. b0) *. u *. nx in
        within
          ~msg:(Printf.sprintf "type %d" kind)
          ~u tol
          (reals
             (Nx.to_array
                (Nx.dct ~type_ ~axes:[ 0 ] (Nx.dct ~type_ ~axes:[ 1 ] x))))
          (reals (Nx.to_array (Nx.dct ~type_ ~axes:[ 0; 1 ] x)))
      end)
    types

(* A narrow format transforms at float32 and rounds once. *)
let test_trig_narrow () =
  let x =
    Nx.create Nx.float64 [| 9 |]
      (Array.init 9 (fun i -> sin (Float.of_int (i * i))))
  in
  let narrow (type s) (dt : (float, s) Nx.dtype) (type_, kind) =
    let h = Nx.cast dt x in
    let wide = Nx.cast Nx.float32 h in
    let msg name = Printf.sprintf "%s type %d of %s" name kind (D.name dt) in
    equal ~msg:(msg "dct") (array float_exact)
      (Nx.to_array (Nx.cast dt (Nx.dct ~type_ wide)))
      (Nx.to_array (Nx.dct ~type_ h));
    equal ~msg:(msg "dst") (array float_exact)
      (Nx.to_array (Nx.cast dt (Nx.dst ~type_ wide)))
      (Nx.to_array (Nx.dst ~type_ h))
  in
  List.iter (narrow Nx.float16) types;
  List.iter (narrow Nx.bfloat16) types

(* Frequencies *)

type float_dtype = Fdt : (float, 's) D.t -> float_dtype

(* Each frequency is [i / (d n)] at float64, [i] wrapped past the middle, stored
   once in the dtype. *)
let law_frequencies (Fdt dt, (n, k)) =
  let d = Float.of_int k /. 8. and f = Float.of_int n in
  cover "a negative spacing" (d < 0.);
  let stored v = Nx.to_array (Nx.create dt [| Array.length v |] v) in
  let i_of j = if j < (n + 1) / 2 then j else j - n in
  equal ~msg:"fftfreq" (array float_exact)
    (stored (Array.init n (fun j -> Float.of_int (i_of j) /. (d *. f))))
    (Nx.to_array (Nx.fftfreq dt ~d n));
  equal ~msg:"rfftfreq" (array float_exact)
    (stored (Array.init ((n / 2) + 1) (fun j -> Float.of_int j /. (d *. f))))
    (Nx.to_array (Nx.rfftfreq dt ~d n))

let frequencies_case =
  Gen.(
    pair
      (of_list
         ~pp:(fun ppf (Fdt dt) -> D.pp ppf dt)
         [ Fdt D.Float64; Fdt D.Float32; Fdt D.Float16 ])
      (pair (int_range 1 64)
         (map (fun k -> if k = 0 then 1 else k) (int_range (-16) 16))))

let test_frequency_rows () =
  equal ~msg:"fftfreq float64 4" (array float_exact)
    [| 0.; 0.25; -0.5; -0.25 |]
    (Nx.to_array (Nx.fftfreq Nx.float64 4));
  equal ~msg:"fftshift (fftfreq float64 5)" (array float_exact)
    [| -0.4; -0.2; 0.; 0.2; 0.4 |]
    (Nx.to_array (Nx.fftshift (Nx.fftfreq Nx.float64 5)));
  equal ~msg:"fftfreq 1" (array float_exact) [| 0. |]
    (Nx.to_array (Nx.fftfreq Nx.float32 1))

module S2 = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])
module S4 = (val Nx.devices (List.init 4 Nx_support.memory))

(* The _at forms are their host forms at their placement, and raise naming
   themselves where it cuts an axis their value lacks. *)
let test_at_forms () =
  let p = S2.split ~axis:0
  and cut1 =
    Nx.Placement.mesh (Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]) [ (1, [ "a" ]) ]
  in
  let check ~msg host at =
    equal ~msg (array float_exact) (Nx.to_array host) (Nx.to_array at);
    equal ~msg:(msg ^ ", where it lies") (Nx_support.placement ()) p
      (Nx.placement at)
  in
  check ~msg:"fftfreq_at"
    (Nx.fftfreq Nx.float32 ~d:0.5 7)
    (Nx.fftfreq_at p Nx.float32 ~d:0.5 7);
  check ~msg:"rfftfreq_at" (Nx.rfftfreq Nx.float32 8)
    (Nx.rfftfreq_at p Nx.float32 8);
  check ~msg:"hann_at" (Nx.hann Nx.float32 8) (Nx.hann_at p Nx.float32 8);
  invalid ~by:"Nx.fftfreq_at" (fun () -> Nx.fftfreq_at cut1 Nx.float32 4);
  invalid ~by:"Nx.rfftfreq_at" (fun () -> Nx.rfftfreq_at cut1 Nx.float32 4);
  invalid ~by:"Nx.hann_at" (fun () -> Nx.hann_at cut1 Nx.float32 4)

(* Shifts *)

(* fftshift rolls each of its axes by [n/2] toward the end: the element at [i]
   along an axis of [n] lies at [(i + n/2) mod n]. *)
let shift_case =
  Gen.(pair (array ~size:(int_range 1 3) (int_range 0 5)) (int_range 0 7))

let axes_of s pick =
  let r = Array.length s in
  let axes =
    List.filter (fun a -> (pick lsr a) land 1 = 1) (List.init r Fun.id)
  in
  if axes = [] then [ r - 1 ] else axes

let law_fftshift (s, pick) =
  let axes = axes_of s pick in
  cover "an odd extent shifted" (List.exists (fun a -> s.(a) mod 2 = 1) axes);
  let n = Array.fold_left ( * ) 1 s in
  let x = Nx.reshape s (Nx.arange Nx.int32 0 n 1) in
  let y = Nx.to_array (Nx.fftshift ~axes x) in
  let want = Array.make n 0l in
  for k = 0 to n - 1 do
    let i = Nx_support.index s k in
    List.iter (fun a -> i.(a) <- (i.(a) + (s.(a) / 2)) mod s.(a)) axes;
    let at = ref 0 in
    Array.iteri (fun a c -> at := (!at * s.(a)) + c) i;
    want.(!at) <- Int32.of_int k
  done;
  equal (array int32) want y

let law_ifftshift (s, pick) =
  let axes = axes_of s pick in
  cover "an odd extent shifted" (List.exists (fun a -> s.(a) mod 2 = 1) axes);
  let x = Nx.reshape s (Nx.arange Nx.int32 0 (Array.fold_left ( * ) 1 s) 1) in
  equal (array int32) (Nx.to_array x)
    (Nx.to_array (Nx.ifftshift ~axes (Nx.fftshift ~axes x)))

(* hann *)

let test_hann_rows () =
  (* Within law_hann's [23u]. *)
  equal ~msg:"hann 4"
    (array (float (23. *. 0x1p-53)))
    [| 0.; 0.5; 1.; 0.5 |]
    (Nx.to_array (Nx.hann Nx.float64 4));
  equal ~msg:"hann 1" (array float_exact) [| 0. |]
    (Nx.to_array (Nx.hann Nx.float32 1));
  equal ~msg:"hann 0" (array int) [| 0 |] (Nx.shape (Nx.hann Nx.float32 0))

(* hann at a narrow dtype is float64's window rounded once; at float64 it is
   within [23u] of [0.5 - 0.5 cos (2πi / n)] computed in OCaml: nx's argument [i
   fl(2π/n)] within [3u t <= 6πu] radians, which [0.5 sin] halves to [9.5u]; cos
   within 2 ulps of the rounded exact value, [2.5u], halved; the subtraction
   [u]: [11.7u]; the reference's the same but libm's cos within an ulp: [10.9u].
   Half a period apart, two points sum to 1 within [25u]: each window's [11.7u]
   and the addition. *)
let law_hann n =
  cover "an even n" (n mod 2 = 0 && n > 0);
  cover "an odd n" (n mod 2 = 1);
  let w = Nx.to_array (Nx.hann Nx.float64 n) in
  let u = 0x1p-53 in
  Array.iteri
    (fun i v ->
      let t = 2. *. pi *. Float.of_int i /. Float.of_int n in
      at_most
        ~msg:(Printf.sprintf "point %d" i)
        float_exact ~than:(23. *. u)
        (Nx_support.gap v (0.5 -. (0.5 *. cos t))))
    w;
  if n mod 2 = 0 then
    for i = 0 to (n / 2) - 1 do
      at_most
        ~msg:(Printf.sprintf "points %d and %d" i (i + (n / 2)))
        float_exact ~than:(25. *. u)
        (Nx_support.gap (w.(i) +. w.(i + (n / 2))) 1.)
    done;
  let rounded (type s) (dt : (float, s) Nx.dtype) =
    equal ~msg:(D.name dt) (array float_exact)
      (Nx.to_array (Nx.cast dt (Nx.hann Nx.float64 n)))
      (Nx.to_array (Nx.hann dt n))
  in
  rounded Nx.float32;
  rounded Nx.float16;
  rounded Nx.bfloat16

(* stft and istft *)

let stft_case = Gen.(triple (int_range 1 9) (int_range 1 9) (int_range 9 40))

let signal_2d n =
  Nx.init Nx.float64 [| 2; n |] (fun i ->
      cos (Float.of_int ((i.(0) * 13) + (i.(1) * 3))))

let bits_of (y : (Complex.t, _, _) Nx.t) =
  Nx.to_array (Nx.bitcast Nx.int64 (Nx.bitcast Nx.float64 y))

(* Frame [f] of [stft] is rfft of the [window] samples from [f step] times
   [win], with its bits: nx.cpu's transform of a line is a function of its
   values whatever its batch (nx_cpu.mli, Fourier transforms). *)
let law_frames (window, step, n) =
  let x = signal_2d n in
  let win = Nx.hann Nx.float64 window in
  let zs = Nx.stft Nx.complex128 ~window ~step ~win x in
  let frames = ((n - window) / step) + 1 in
  cover "several frames" (frames > 1);
  equal ~msg:"shape" (array int) [| 2; frames; (window / 2) + 1 |] (Nx.shape zs);
  for f = 0 to frames - 1 do
    let frame = Nx.slice [ Nx.A; Nx.R (f * step, (f * step) + window) ] x in
    equal
      ~msg:(Printf.sprintf "frame %d" f)
      (array int64)
      (bits_of (Nx.rfft Nx.complex128 (Nx.mul frame win)))
      (bits_of (Nx.copy (Nx.slice [ Nx.A; Nx.I f; Nx.A ] zs)))
  done

(* stft's step defaults to [max 1 (window / 4)] and its taper to Hann, as
   istft's do. *)
let test_defaults () =
  let x = signal_2d 20 in
  List.iter
    (fun window ->
      let step = Stdlib.max 1 (window / 4) in
      let win = Nx.hann Nx.float64 window in
      let z = Nx.stft Nx.complex128 ~window x in
      let msg = Printf.sprintf "window %d" window in
      equal ~msg:(msg ^ ", stft") (array int64)
        (bits_of (Nx.stft Nx.complex128 ~window ~step ~win x))
        (bits_of z);
      equal ~msg:(msg ^ ", istft") (array float_exact)
        (Nx.to_array (Nx.istft Nx.float64 ~window ~step ~win z))
        (Nx.to_array (Nx.istft Nx.float64 ~window z)))
    [ 1; 3; 4; 9 ]

(* istft (stft x) is x where a frame's taper holds the sample, and [0.] where
   none does; [length] cuts it. Each frame's round trip is within the two
   transforms' bound, [√2 (2ε + 6) u ‖frame w‖₂] (rfft, then irfft whose operand
   norm is √2 its bins'); a sample adds its frames' errors times their taper
   there, divides by the sum [E] of the tapers' squares there, and takes [F + 2]
   roundings of [|x|] for its [F] frames' sum and the division. *)
let law_stft (window, step, n) =
  let step = Stdlib.min step window in
  let x = signal_2d n in
  let xs = Nx.to_array x in
  let frames = ((n - window) / step) + 1 in
  let covered = ((frames - 1) * step) + window in
  let u = 0x1p-53 in
  let check ~msg w y =
    let ys = Nx.to_array y in
    for row = 0 to 1 do
      let frame_err f =
        let s = ref 0. in
        for j = 0 to window - 1 do
          let v = xs.((row * n) + (f * step) + j) *. w.(j) in
          s := !s +. (v *. v)
        done;
        Float.sqrt 2. *. ((2. *. e window) +. 6.) *. u *. Float.sqrt !s
      in
      for i = 0 to covered - 1 do
        let held = ref 0 and spread = ref 0. and envelope = ref 0. in
        for f = 0 to frames - 1 do
          let j = i - (f * step) in
          if j >= 0 && j < window then begin
            incr held;
            spread := !spread +. (Float.abs w.(j) *. frame_err f);
            envelope := !envelope +. (w.(j) *. w.(j))
          end
        done;
        let x_i = xs.((row * n) + i) in
        if !envelope = 0. then
          equal
            ~msg:(Printf.sprintf "%s, sample %d, held by no taper" msg i)
            float_exact 0.
            ys.((row * covered) + i)
        else
          at_most
            ~msg:(Printf.sprintf "%s, sample %d" msg i)
            float_exact
            ~than:
              ((!spread /. !envelope)
              +. (Float.of_int (!held + 2) *. u *. Float.abs x_i))
            (Nx_support.gap ys.((row * covered) + i) x_i)
      done
    done
  in
  let ones = Array.make window 1. in
  let win1 = Nx.ones Nx.float64 [| window |] in
  check ~msg:"untapered" ones
    (Nx.istft Nx.float64 ~window ~step ~win:win1
       (Nx.stft Nx.complex128 ~window ~step ~win:win1 x));
  let z = Nx.stft Nx.complex128 ~window ~step x in
  let y = Nx.istft Nx.float64 ~window ~step z in
  check ~msg:"tapered" (Nx.to_array (Nx.hann Nx.float64 window)) y;
  let k = (covered / 2) + 1 in
  cover "length cuts" (k < covered);
  equal ~msg:"length cuts" (array float_exact)
    (Nx.to_array (Nx.slice [ Nx.A; Nx.R (0, k) ] y))
    (Nx.to_array (Nx.istft Nx.float64 ~window ~step ~length:k z))

(* Refusals *)

let test_refusals () =
  let z = Nx.zeros Nx.complex64 [| 4 |] and x = Nx.zeros Nx.float32 [| 4 |] in
  let z2 = Nx.zeros Nx.complex64 [| 2; 3 |] in
  invalid ~by:"Nx.fft" (fun () -> Nx.fft (Nx.zeros Nx.complex64 [||]));
  invalid ~by:"Nx.fft" (fun () -> Nx.fft ~axes:[] z);
  invalid ~by:"Nx.fft" (fun () -> Nx.fft ~axes:[ 1 ] z);
  invalid ~by:"Nx.ifft" (fun () ->
      Nx.ifft ~axes:[ 0; 0 ] (Nx.zeros Nx.complex64 [| 2; 2 |]));
  invalid ~by:"Nx.fft" (fun () -> Nx.fft ~s:[ 1; 2 ] z);
  invalid ~by:"Nx.fft" (fun () -> Nx.fft ~s:[ -1 ] z);
  invalid ~by:"Nx.rfft" (fun () -> Nx.rfft Nx.complex64 ~s:[ -1 ] x);
  invalid ~by:"Nx.irfft" (fun () ->
      Nx.irfft Nx.float32 (Nx.zeros Nx.complex64 [| 0 |]));
  invalid ~by:"Nx.hfft" (fun () ->
      Nx.hfft Nx.float32 (Nx.zeros Nx.complex64 [| 0 |]));
  invalid ~by:"Nx.hfft" (fun () -> Nx.hfft Nx.float32 ~axes:[ 1 ] z);
  invalid ~by:"Nx.ihfft" (fun () -> Nx.ihfft Nx.complex64 ~s:[ -1 ] x);
  invalid ~by:"Nx.dct" (fun () ->
      Nx.dct ~type_:Type_1 (Nx.zeros Nx.float32 [| 1 |]));
  invalid ~by:"Nx.idct" (fun () ->
      Nx.idct ~type_:Type_1 (Nx.zeros Nx.float32 [| 0; 1 |]));
  invalid ~by:"Nx.dst" (fun () -> Nx.dst ~axes:[ 1 ] x);
  invalid ~by:"Nx.idst" (fun () -> Nx.idst ~axes:[ 0; 0 ] x);
  invalid ~by:"Nx.fftfreq" (fun () -> Nx.fftfreq Nx.float32 0);
  invalid ~by:"Nx.fftfreq" (fun () -> Nx.fftfreq Nx.float32 ~d:0. 4);
  invalid ~by:"Nx.rfftfreq" (fun () -> Nx.rfftfreq Nx.float32 0);
  invalid ~by:"Nx.rfftfreq" (fun () -> Nx.rfftfreq Nx.float32 ~d:Float.nan 4);
  invalid ~by:"Nx.fftshift" (fun () -> Nx.fftshift (Nx.zeros Nx.float32 [||]));
  invalid ~by:"Nx.fftshift" (fun () -> Nx.fftshift ~axes:[] x);
  invalid ~by:"Nx.ifftshift" (fun () -> Nx.ifftshift ~axes:[ 0; 0 ] x);
  invalid ~by:"Nx.hann" (fun () -> Nx.hann Nx.float32 (-1));
  invalid ~by:"Nx.stft" (fun () ->
      Nx.stft Nx.complex64 ~window:2 (Nx.zeros Nx.float32 [||]));
  invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:0 x);
  invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:2 ~step:0 x);
  invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:5 x);
  invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:2 ~win:x x);
  invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 z);
  invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:0 z2);
  invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 ~step:5 z2);
  invalid ~by:"Nx.istft" (fun () ->
      Nx.istft Nx.float32 ~window:4 (Nx.zeros Nx.complex64 [| 2; 4 |]));
  invalid ~by:"Nx.istft" (fun () ->
      Nx.istft Nx.float32 ~window:4 ~win:(Nx.zeros Nx.float32 [| 3 |]) z2);
  invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 ~length:0 z2)

(* A composite's refusal from a function it calls names the composite: a taper
   on another device than the signal. *)
let test_composites_named () =
  let on k x = Nx.place (Nx.Placement.device S2.v (Nx_support.memory k)) x in
  let x = on 0 (Nx.zeros Nx.float32 [| 8 |]) in
  let win = on 1 (Nx.ones Nx.float32 [| 4 |]) in
  invalid ~by:"Nx.stft" (fun () -> Nx.stft Nx.complex64 ~window:4 ~win x);
  let z = on 0 (Nx.zeros Nx.complex64 [| 2; 3 |]) in
  invalid ~by:"Nx.istft" (fun () -> Nx.istft Nx.float32 ~window:4 ~win z)

(* Prim.Fft *)

module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let on a = Nx.place Count.on (Nx.Repr.of_array Nx.Host.v a)
let fft_op f = Nx.Prim.eval ~by:"t" (Fft f)

(* Values exact in the sum, each within nx.cpu's bound. *)
let near ~msg ~u n want got =
  within ~msg ~u
    (allowance ~u ~c:1. ~eps:(e n) ~points:(Float.of_int n) ~n:1
       ~norm:(norm2 want) ~abs:0.)
    want got

let test_c2c () =
  let x =
    on
      (A.of_array D.Complex128 [| 4 |] [| z 1. 0.; z 2. 0.; z 3. 0.; z 4. 0. |])
  in
  let y = fft_op (C2c { direction = Forward; axes = [| 0 |]; x }) in
  near ~msg:"forward" ~u:0x1p-53 4
    [| z 10. 0.; z (-2.) 2.; z (-2.) 0.; z (-2.) (-2.) |]
    (read y);
  near ~msg:"inverse, undivided" ~u:0x1p-53 4
    [| z 4. 0.; z 8. 0.; z 12. 0.; z 16. 0. |]
    (read (fft_op (C2c { direction = Inverse; axes = [| 0 |]; x = y })));
  let impulse =
    on
      (A.of_array D.Complex128 [| 2; 3 |]
         (Array.init 6 (fun i -> z (if i = 0 then 1. else 0.) 0.)))
  in
  near ~msg:"an impulse over both axes" ~u:0x1p-53 6
    (Array.make 6 (z 1. 0.))
    (read
       (fft_op (C2c { direction = Forward; axes = [| 0; 1 |]; x = impulse })))

let test_r2c_c2r () =
  let x = on (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  let y = fft_op (R2c { dtype = Nx.complex64; axes = [| 0 |]; x }) in
  near ~msg:"r2c: bins 0 to n/2" ~u:0x1p-24 4
    [| z 10. 0.; z (-2.) 2.; z (-2.) 0. |]
    (read y);
  let back =
    fft_op (C2r { dtype = Nx.float32; n = 4; axes = [| 0 |]; x = y })
  in
  near ~msg:"c2r of 4 points, undivided" ~u:0x1p-24 4
    (reals [| 4.; 8.; 12.; 16. |])
    (reals (read_real back));
  let five =
    fft_op (C2r { dtype = Nx.float32; n = 5; axes = [| 0 |]; x = y })
  in
  let full = hermitian (read y) 5 in
  within ~msg:"c2r of 5 points" ~u:0x1p-24
    (allowance ~u:0x1p-24 ~c:1. ~eps:(e 5) ~points:5. ~n:5
       ~norm:(Float.sqrt 2. *. norm2 full)
       ~abs:(sum_abs full))
    (reals (Array.map (fun (v : Complex.t) -> v.re) (dft ~sign:1. full)))
    (reals (read_real five))

(* A dtype of another precision than [x]'s, a narrow float, bins too few for
   [n], an empty or outside axis raise before any kernel is called. *)
let test_rules () =
  let x = on (A.of_array D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  let c = on (A.of_array D.Complex64 [| 3 |] (Array.make 3 Complex.zero)) in
  let x16 = Nx.cast Nx.float16 x in
  C.reset ();
  let raises f = raises_match (Exn.invalid_arg ~sub:"t: ") f in
  raises (fun () -> fft_op (R2c { dtype = Nx.complex128; axes = [| 0 |]; x }));
  raises (fun () ->
      fft_op (R2c { dtype = Nx.complex64; axes = [| 0 |]; x = x16 }));
  raises (fun () ->
      fft_op (C2r { dtype = Nx.float32; n = 6; axes = [| 0 |]; x = c }));
  raises (fun () ->
      fft_op (C2r { dtype = Nx.float64; n = 4; axes = [| 0 |]; x = c }));
  raises (fun () -> fft_op (C2c { direction = Forward; axes = [||]; x = c }));
  raises (fun () -> fft_op (C2c { direction = Forward; axes = [| 1 |]; x = c }));
  equal ~msg:"kernel calls" int 0 (C.calls ())

(* Transforms at chosen frequencies: zoom_fft against the direct sum at float64
   of its definition, each phase reduced modulo one cycle exactly but for one
   rounding, within the bound its doc states for nx.cpu plus the reference's own
   error. *)
module Zoom = struct
  let u64 = 0x1p-53

  (* [a b] as [hi + lo], exactly. *)
  let two_product a b =
    let hi = a *. b in
    [ hi; Float.fma a b (-.hi) ]

  let centred x = x -. Float.round x

  (* The sum modulo one of exact terms, each first centred exactly, by
     Neumaier's compensated sum: within [u/2] of the exact phase, to first
     order. *)
  let cycles terms =
    let s = ref 0. and c = ref 0. in
    List.iter
      (fun t ->
        let t = centred t in
        let r = !s +. t in
        if Float.abs !s >= Float.abs t then c := !c +. (!s -. r +. t)
        else c := !c +. (t -. r +. !s);
        s := centred r)
      terms;
    centred (!s +. !c)

  (* [y[k] = Σ_j x[j] e^(-2πi j f_k)], [f_k = start + k step], with the phase [j
     start + (j k) step] of exact products, summed part by part by Neumaier's
     sum. [j] and [j k] are integers, so [start] and [step] are first reduced
     modulo one, exactly. *)
  let direct ~start ~step m (x : Complex.t array) =
    let start = Float.rem start 1. and step = Float.rem step 1. in
    Array.init m (fun k ->
        let re = ref 0. and re_c = ref 0. and im = ref 0. and im_c = ref 0. in
        let add acc c t =
          let r = !acc +. t in
          if Float.abs !acc >= Float.abs t then c := !c +. (!acc -. r +. t)
          else c := !c +. (t -. r +. !acc);
          acc := r
        in
        Array.iteri
          (fun j (v : Complex.t) ->
            let j' = Float.of_int j in
            let r =
              cycles
                (two_product start j' @ two_product step (Float.of_int (j * k)))
            in
            let t = -2. *. Float.pi *. r in
            let c = cos t and s = sin t in
            add re re_c ((v.re *. c) -. (v.im *. s));
            add im im_c ((v.re *. s) +. (v.im *. c)))
          x;
        { Complex.re = !re +. !re_c; im = !im +. !im_c })

  (* The reference's error at each [k], in units of [u64] times [Σ |x|]: its
     phase within [u/2] cycle and [2π r]'s two roundings, about [2π u] radians;
     [cos] and [sin] within an ulp each; the product's roundings. *)
  let reference_error = (2. *. Float.pi) +. 6.

  (* nx.cpu's [e(L)] for [L = 2^p] points (Nx_cpu, Fourier transforms). *)
  let cpu_e p = (8. *. Float.of_int p *. Float.pow 2. 1.5) +. 8.

  (* The chirps' error [δ] in units of [u]: one rounding of a float64 chirp into
     complex64's parts, or, for complex128, the float64 evaluation's: the phase
     within [u/2] cycle, [2π r]'s roundings, and nx.cpu's [cos] and [sin] within
     2 ulps each. *)
  let delta ~wide = if wide then (2.5 *. Float.pi) +. (2. *. sqrt 2.) else 1.
  let norm a = sqrt (Array.fold_left (fun s z -> s +. Complex.norm2 z) 0. a)

  (* The worst-case bound of zoom_fft's doc, [(2ε + ε' + 4δ + 3√5 + 1) (n + m -
     1) u ‖x‖₂] with [ε = ε'] nx.cpu's constant for [L] points, plus the
     reference's error over [m] points. *)
  let allowance ~wide n m x =
    let rec log2 l p = if l >= n + m - 1 then p else log2 (2 * l) (p + 1) in
    let e = cpu_e (log2 1 0) in
    let u = if wide then u64 else 0x1p-24 in
    let k = Float.of_int (n + m - 1) in
    let sum = Array.fold_left (fun s z -> s +. Complex.norm z) 0. x in
    ((3. *. e) +. (4. *. delta ~wide) +. (3. *. sqrt 5.) +. 1.)
    *. k *. u *. norm x
    +. (reference_error *. u64 *. sqrt (Float.of_int m) *. sum)

  type cdt = Cdt : (Complex.t, 's) D.t * bool -> cdt

  let dtypes =
    Gen.of_list
      ~pp:(fun ppf (Cdt (dt, _)) -> D.pp ppf dt)
      [ Cdt (D.Complex64, false); Cdt (D.Complex128, true) ]

  (* A signal rounded into [dt]'s parts, read back at float64. *)
  let signal (type s) (dt : (Complex.t, s) D.t) s parts =
    let x = Nx.create dt s parts in
    (x, Nx.to_array (Nx.cast Nx.complex128 x))

  (* The lines of [a], of shape [s], along [axis]. *)
  let lines s axis (a : 'a array) =
    let r = Array.length s in
    let inner =
      Array.fold_left ( * ) 1 (Array.sub s (axis + 1) (r - axis - 1))
    in
    let outer = Array.fold_left ( * ) 1 (Array.sub s 0 axis) in
    let n = s.(axis) in
    List.concat
      (List.init outer (fun o ->
           List.init inner (fun i ->
               Array.init n (fun j -> a.((((o * n) + j) * inner) + i)))))

  let check (Cdt (dt, wide)) ~start ~step ~axis m s parts =
    let x, xs = signal dt s parts in
    let y = Nx.zoom_fft ~axis ~start ~step m x in
    let out = Array.copy s in
    out.(axis) <- m;
    equal ~msg:"shape" (array int) out (Nx.shape y);
    let ys = Nx.to_array (Nx.cast Nx.complex128 y) in
    List.iter2
      (fun x y ->
        let expected = direct ~start ~step m x in
        let diff = distance expected y in
        at_most ~msg:"‖ŷ - y‖₂" float_exact
          ~than:(allowance ~wide s.(axis) m x)
          diff)
      (lines s axis xs) (lines out axis ys)

  let pp_float ppf v = Format.fprintf ppf "%h" v

  let case =
    let open Gen in
    let frequency =
      one_of
        [
          float_range (-1.) 1.;
          float_range (-1e6) 1e6;
          of_list ~pp:pp_float
            [
              0.; 0.5; 1. /. 3.; -0.25; -7.5; 1e300; Float.max_float; -0x1p1023;
            ];
        ]
    in
    (* One point, one frequency and the steps 1/2 and 1/3 in about half the
       cases each, so that every run meets each cover. *)
    let points = one_of [ constant 1; int_range 1 40 ] in
    let step = one_of [ of_list ~pp:pp_float [ 0.5; 1. /. 3. ]; frequency ] in
    (* Parts of magnitudes from [2^-20], so that no square underflows: the bound
       holds barring underflow. *)
    let part =
      map
        (fun (neg, m) -> if neg then -.m else m)
        (pair bool (float_range 0x1p-20 1.))
    in
    let* dt = dtypes in
    let* n = points and+ m = points in
    let* lead = bool in
    let* axis = if lead then int_range 0 1 else constant 0 in
    let s =
      if not lead then [| n |] else if axis = 0 then [| n; 2 |] else [| 2; n |]
    in
    let* start = frequency and+ step = step in
    let+ parts =
      array
        ~size:(constant (Array.fold_left ( * ) 1 s))
        (map (fun (re, im) -> { Complex.re; im }) (pair part part))
    in
    (dt, n, m, s, axis, start, step, parts)

  (* A start past 2^53 and a step of 1/2, over 3 points to 2 frequencies: the
     cover a draw reaches 0.18 of the time. *)
  let far =
    ( Cdt (D.Complex128, true),
      3,
      2,
      [| 3 |],
      0,
      1e300,
      0.5,
      [|
        { Complex.re = 0.5; im = -0.25 };
        { re = 1.; im = 0.125 };
        { re = -0.75; im = 1. };
      |] )

  let law_direct =
    prop ~examples:[ far ] "zoom_fft is the direct sum within its bound"
      (Gen.with_pp
         (fun ppf (Cdt (dt, _), _, m, s, axis, start, step, _) ->
           Format.fprintf ppf "%a %s axis %d, m %d, start %h, step %h" D.pp dt
             (String.concat "x" (Array.to_list (Array.map string_of_int s)))
             axis m start step)
         case)
      (fun (dt, n, m, s, axis, start, step, parts) ->
        cover "one point" (n = 1);
        cover "one frequency" (m = 1);
        cover "a leading axis" (axis = 0 && Array.length s = 2);
        cover "a step of 1/2 or 1/3" (step = 0.5 || step = 1. /. 3.);
        cover "a frequency past 2^53"
          (Float.abs start > 0x1p53 || Float.abs step > 0x1p53);
        check dt ~start ~step ~axis m s parts)

  (* Where [step] has a small denominator [‖H‖_∞] nears [n + m - 1], the bound's
     worst case. *)
  let test_bound () =
    let parts n =
      Array.init n (fun j ->
          let t = Float.of_int j in
          { Complex.re = sin (t *. 0.7) +. 0.25; im = cos (t *. 1.3) })
    in
    List.iter
      (fun dt ->
        List.iter
          (fun (start, step) ->
            check dt ~start ~step ~axis:0 200 [| 300 |] (parts 300))
          [ (0., 0.5); (0.1, 0.5); (0., 1. /. 3.); (0.3, 1. /. 3.) ])
      [ Cdt (D.Complex64, false); Cdt (D.Complex128, true) ]

  (* With [start = 0], [step = 1 / n] and [m = n], fft's transform. *)
  let test_fft () =
    let n = 12 in
    let parts =
      Array.init n (fun j ->
          {
            Complex.re = Float.of_int (j mod 5);
            im = Float.of_int (j mod 3) -. 1.;
          })
    in
    let x = Nx.create Nx.complex128 [| n |] parts in
    let y = Nx.to_array (Nx.zoom_fft ~start:0. ~step:(1. /. 12.) n x) in
    let f = Nx.to_array (Nx.fft x) in
    (* Both bounds: zoom_fft's for L = 32 and fft's for 12 points, e(12) = 8 (2
       · 2^1.5 + 3^1.5) + 8. *)
    let e12 = (8. *. ((2. *. Float.pow 2. 1.5) +. Float.pow 3. 1.5)) +. 8. in
    let bound =
      allowance ~wide:true n n parts
      +. (e12 *. u64 *. sqrt (Float.of_int n) *. norm parts)
    in
    at_most float_exact ~than:bound (distance f y)

  let test_empty () =
    let x = Nx.create Nx.complex64 [| 2; 0 |] [||] in
    Nx_support.equal_elements D.Complex64
      (Array.make 6 Complex.zero)
      (Nx.to_array (Nx.zoom_fft ~start:0.1 ~step:0.2 3 x));
    let x = Nx.create Nx.complex64 [| 2; 3 |] (Array.make 6 Complex.one) in
    equal (array int) [| 2; 0 |]
      (Nx.shape (Nx.zoom_fft ~start:0.1 ~step:0.2 0 x))

  let test_refusals () =
    let by = "Nx.zoom_fft" in
    let invalid f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f in
    let x = Nx.create Nx.complex64 [| 3 |] (Array.make 3 Complex.one) in
    invalid (fun () -> Nx.zoom_fft ~start:0. ~step:0.1 (-1) x);
    invalid (fun () -> Nx.zoom_fft ~axis:1 ~start:0. ~step:0.1 2 x);
    invalid (fun () -> Nx.zoom_fft ~start:Float.nan ~step:0.1 2 x);
    invalid (fun () -> Nx.zoom_fft ~start:0. ~step:Float.infinity 2 x);
    invalid (fun () ->
        Nx.zoom_fft ~start:0. ~step:0.1 2
          (Nx.create Nx.complex64 [||] [| Complex.one |]))

  (* Chirps past [j = 9.49e7], where [j²] leaves float64's integers: each is
     within [δ] of [e^(-2πi (start j + step j² / 2))], whose phase is exact here
     in integers: a float [a = M 2^-K] has [a J mod 1] = [(M J mod 2^K) / 2^K],
     and OCaml's products keep [M J]'s low 63 bits. *)
  let exact_cycles a big =
    let f, e = Float.frexp a in
    let k = 53 - e in
    assert (k >= 0 && k <= 62);
    let mant = Float.to_int (Float.ldexp f 53) in
    let r = mant * big land ((1 lsl k) - 1) in
    Float.ldexp (Float.of_int r) (-k)

  let test_large () =
    List.iter
      (fun (start, step, first, count) ->
        let c =
          Nx.to_array (Nx.zoom_chirp D.Complex128 ~start ~step ~first count)
        in
        Array.iteri
          (fun i (got : Complex.t) ->
            let j = first + i in
            let r =
              centred (exact_cycles start j +. exact_cycles (step /. 2.) (j * j))
            in
            let t = -2. *. Float.pi *. r in
            let d =
              distance [| { Complex.re = cos t; im = sin t } |] [| got |]
            in
            (* [δ] and the reference's own error. *)
            let tol = (delta ~wide:true +. reference_error +. 2.) *. u64 in
            at_most ~msg:(Printf.sprintf "j = %d" j) float_exact ~than:tol d)
          c)
      [
        (0.3, 1. /. 3., 94_906_260, 16);
        (-0.3, 0.999, 94_906_260, 16);
        (0.1, 1. /. 3., 2_000_000_000, 8);
        (0.25, 12345.678, 1_500_000_001, 8);
      ]
end

let () =
  exit
    (run "nx fft"
       [
         group "fft ifft"
           [
             prop ~examples:short
               "fft and ifft are the sums within their bound, ifft (fft x) is \
                x, under each norm"
               signal law_fft;
             prop
               "along two axes, the transform along each in turn, scaled over \
                their points"
               Gen.(
                 pair
                   (of_list ~pp:pp_precision precisions)
                   (pair (int_range 1 4) (int_range 1 5)))
               law_axes;
             test "s cuts or extends the operand with zeros" test_sizes;
             test
               "the scaling divides each part, an infinite one keeps its zero"
               test_scaled_parts;
           ];
         group "zoom_fft"
           [
             Zoom.law_direct;
             test
               "within its bound at steps 1/2 and 1/3, where its chirp's \
                transform nears n + m - 1"
               Zoom.test_bound;
             test "at start 0, step 1/n and n points it is fft's transform"
               Zoom.test_fft;
             test "of no point is zeros, to no frequency empty" Zoom.test_empty;
             test "raises on m < 0, an axis outside, a start or step not finite"
               Zoom.test_refusals;
             test "its chirps past j = 9.49e7 are within their bound"
               Zoom.test_large;
           ];
         group "rfft irfft"
           [
             prop ~examples:short
               "rfft is fft's first n/2 + 1 bins and irfft (rfft x) is x, \
                under each norm"
               signal law_real;
             prop "irfft is its linear map, Hermitian operand or not"
               Gen.(pair (int_range 1 7) (int_range 1 10))
               law_irfft;
             test
               "irfft reads no imaginary part of bin 0 or, at an even n, of \
                bin n/2"
               test_unread_parts;
             test "a narrow float transforms at float32, rounded once"
               test_narrow;
             test
               "a transform computes at the wider precision, rounded once into \
                its dtype"
               test_wider;
           ];
         group "hfft ihfft"
           [
             prop ~examples:one_bin
               "hfft is the forward sum of the Hermitian signal, ihfft its \
                inverse"
               bins_and_length law_hermitian;
             test "over two axes in either order" test_hermitian_axes;
           ];
         group "dct dst"
           [
             prop ~examples:trig_short
               "each type is its sum within its bound, Forward divides by the \
                logical length, and the inverse undoes it under each norm"
               trig_signal law_trig;
             prop "along two axes, the transform along each in turn"
               Gen.(pair (int_range 1 5) (int_range 1 5))
               law_trig_axes;
             test "a narrow float transforms at float32, rounded once"
               test_trig_narrow;
             cases "an axis of no element gives a copy of the shape"
               ~id:(fun (s, _) -> Format.asprintf "%a" Nx_support.pp_ints s)
               [
                 ([| 0 |], [ 0 ]);
                 ([| 0; 3 |], [ 0; 1 ]);
                 ([| 3; 0 |], [ 1 ]);
                 ([| 2; 0 |], [ 0 ]);
               ]
               empty_trig;
           ];
         group "fftfreq rfftfreq"
           [
             test "the doc's frequencies" test_frequency_rows;
             prop "each is i / (d n) at float64, stored in its dtype"
               frequencies_case law_frequencies;
             test "the _at forms are the host forms at their placement"
               test_at_forms;
           ];
         group "fftshift ifftshift"
           [
             prop "fftshift rolls each axis by n/2 toward the end" shift_case
               law_fftshift;
             prop "ifftshift undoes fftshift" shift_case law_ifftshift;
           ];
         group "hann"
           [
             test "the formula's rows" test_hann_rows;
             prop ~examples:[ 0; 1; 2 ]
               "float64's window within its bound, rounded once into a narrow \
                dtype, its half-shifted points summing to 1"
               Gen.(int_range 0 64)
               law_hann;
           ];
         group "stft istft"
           [
             test "step and taper default to max 1 (window / 4) and Hann"
               test_defaults;
             prop "each frame is rfft of the frame times win" stft_case
               law_frames;
             prop "istft (stft x) is x where a taper holds it, within its bound"
               stft_case law_stft;
           ];
         group "refusals"
           [
             test "each refusal names its function" test_refusals;
             test "a composite names its function" test_composites_named;
           ];
         group "Prim.Fft"
           [
             test "C2c is the unscaled sum along each of axes" test_c2c;
             test "R2c keeps bins 0 to n/2 and C2r inverts them unscaled"
               test_r2c_c2r;
             test "the rule raises before any kernel" test_rules;
           ];
       ])
