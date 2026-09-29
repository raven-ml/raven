(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Transforms against a discrete Fourier transform written as a loop on
   complex128, each result within 8 (n + 1) eps times the magnitude of what it
   sums; inverses, norms, shifts and windows by the identities that define
   them. *)

open Windtrap
open Nx_test

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let pp_complex ppf (z : Complex.t) = Format.fprintf ppf "%.17g%+.17gi" z.re z.im

let pp_ints =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
    Format.pp_print_int

let pp_norm ppf n =
  Format.pp_print_string ppf
    (match n with
    | `Backward -> "`Backward"
    | `Forward -> "`Forward"
    | `Ortho -> "`Ortho")

let pp_option pp ppf = function
  | None -> Format.pp_print_string ppf "default"
  | Some v -> pp ppf v

let c re im = { Complex.re; im }
let conj (z : Complex.t) = c z.re (-.z.im)
let scale s (z : Complex.t) = c (s *. z.re) (s *. z.im)
let of_real x = c x 0.
let re (z : Complex.t) = z.re
let cmag (z : Complex.t) = Float.abs z.re +. Float.abs z.im
let finite (z : Complex.t) = Float.is_finite z.re && Float.is_finite z.im

(* A storage precision: the gap between one and the next float, and the least
   positive float, the rounding of what underflows. *)
type precision = { eps : float; tiny : float }

let f64 = { eps = epsilon_float; tiny = 0x1p-1074 }
let f32 = { eps = 0x1p-23; tiny = 0x1p-149 }
let f16 = { eps = 0x1p-10; tiny = 0x1p-24 }
let bf16 = { eps = 0x1p-7; tiny = 0x1p-133 }
let eps64 = f64.eps

(* Comparison *)

(* NaN equals NaN: a transform of an empty axis scaled by 1/0 gives NaN on both
   sides. *)
let near tol a b =
  (Float.is_nan a && Float.is_nan b) || a = b || Float.abs (a -. b) <= tol

let cnear tol (a : Complex.t) (b : Complex.t) =
  near tol a.re b.re && near tol a.im b.im

(* An expected value with the distance within which an actual value equals it;
   the actual side carries NaN for a distance. *)
let within pp near =
  Testable.make
    ~pp:(fun ppf (v, tol) ->
      if Float.is_nan tol then pp ppf v
      else Format.fprintf ppf "%a ±%.1e" pp v tol)
    ~equal:(fun (e, tol) (a, _) -> near tol e a)

let reals = within pp_float near
let complexes = within pp_complex cnear

(* Each element of [actual] within its element of [bound], which broadcasts, of
   [expected]. *)
let agrees ?__POS__ w expected bound actual =
  equal ?__POS__ (Ref.witness w)
    (Ref.map2 (fun v tol -> (v, tol)) expected bound)
    (Ref.map (fun v -> (v, Float.nan)) actual)

(* The bound on a transform over [points] points of [x] along [axes]: eight
   times [points + 1] roundings of the magnitude each output sums, scaled as the
   output is, or of what underflows. It keeps the transformed axes as ones, to
   broadcast. *)
let bound ?(prec = f64) ?(scale = 1.) ~axes ~points mag x =
  Ref.map
    (fun s ->
      8. *. Float.of_int (points + 1) *. ((prec.eps *. scale *. s) +. prec.tiny))
    (Ref.reduce ~axes ~keepdims:true ( +. ) 0. (Ref.map mag x))

let worst (b : float Ref.t) = Array.fold_left Float.max 0. b.data
let cclose tol = Testable.make ~pp:pp_complex ~equal:(cnear tol)
let rclose tol = Testable.make ~pp:pp_float ~equal:(near tol)

(* Reference transforms *)

(* The bins [ks] of the unscaled DFT of [x], sign -1 forward and +1 inverse.
   Each twiddle comes from its index reduced mod n, so the reference rounds only
   in its sums. *)
let dft_at ~sign (x : Complex.t array) ks =
  let n = Array.length x in
  let angle m = 2. *. Float.pi *. Float.of_int m /. Float.of_int n in
  let cs = Array.init n (fun m -> Float.cos (angle m)) in
  let sn = Array.init n (fun m -> Float.of_int sign *. Float.sin (angle m)) in
  Array.map
    (fun k ->
      let r = ref 0. and i = ref 0. in
      for j = 0 to n - 1 do
        let m = k * j mod n and xr = x.(j).re and xi = x.(j).im in
        r := !r +. (xr *. cs.(m)) -. (xi *. sn.(m));
        i := !i +. (xr *. sn.(m)) +. (xi *. cs.(m))
      done;
      c !r !i)
    ks

let dft ~sign x = dft_at ~sign x (Array.init (Array.length x) Fun.id)

let resize n zero lane =
  Array.init n (fun i -> if i < Array.length lane then lane.(i) else zero)

(* The factor a transform over [n] points applies under [norm]. *)
let factor ~inverse norm n =
  let n = Float.of_int n in
  match (Option.value norm ~default:`Backward, inverse) with
  | `Backward, false | `Forward, true -> 1.
  | `Backward, true | `Forward, false -> 1. /. n
  | `Ortho, _ -> 1. /. Float.sqrt n

(* The DFT along each of [axes] in turn, each cropped or zero-padded to its size
   first. *)
let dft_along ~sign ~axes ~sizes x =
  List.fold_left2
    (fun x axis n ->
      Ref.along ~axis ~length:n (fun l -> dft ~sign (resize n Complex.zero l)) x)
    x axes sizes

(* The first n/2 + 1 bins of the DFT of [lane] cropped or zero-padded to n. *)
let half_spectrum n lane =
  dft_at ~sign:(-1)
    (Array.map of_real (resize n 0. lane))
    (Array.init ((n / 2) + 1) Fun.id)

(* The length-[n] spectrum whose bins up to n/2 are [g], cropped or zero-padded,
   the others their mirrored conjugates, with the imaginary parts of DC and of
   an even length's Nyquist bin dropped. *)
let hermitian n g =
  let g = resize ((n / 2) + 1) Complex.zero g in
  Array.init n (fun k ->
      if k = 0 || 2 * k = n then c g.(k).re 0.
      else if k <= n / 2 then g.(k)
      else conj g.(n - k))

let product = List.fold_left ( * ) 1

(* Generators *)

let values =
  Gen.frequency
    [
      (8, Gen.float_range (-1.) 1.);
      ( 1,
        Gen.of_list ~pp:pp_float
          [ 0.; -0.; 1e3; -1e3; 5e-324; 2.2250738585072014e-308 ] );
    ]

let complex_values =
  Gen.with_pp pp_complex
    (Gen.map (fun (a, b) -> c a b) (Gen.pair values values))

let non_finite =
  Gen.frequency
    [
      (6, values);
      (1, Gen.of_list ~pp:pp_float [ Float.nan; infinity; neg_infinity ]);
    ]

(* Lengths either side of the transforms' paths: every radix (2, 3, 4, 5, 7, 8),
   the generic passes (11, 13), Bluestein's primes and composites (17, 19, 34 =
   2 x 17, 289), and odd lengths the real transforms take whole. *)
let notable =
  List.concat
    [
      [ 1; 2; 3; 4; 5; 7; 8; 9; 16; 25; 27; 32; 49; 64; 100; 125; 128; 243 ];
      [ 256; 343; 512; 11; 13; 26; 121; 143; 169 ];
      [ 17; 19; 31; 34; 97; 127; 289 ];
    ]

let length =
  Gen.frequency
    [
      (3, Gen.int_range 1 40); (2, Gen.of_list ~pp:Format.pp_print_int notable);
    ]

(* A tensor with one axis of a drawn length among up to two short ones, under a
   drawn layout, and an axis to transform, mostly the long one, counted from
   either end. *)
let signal ?(length = length) ~pp dtype value =
  let drawn =
    let open Gen in
    let* n = length in
    let* rank = int_range 1 3 in
    let* at = int_range 0 (rank - 1) in
    let* short =
      array ~size:(constant (rank - 1)) (int_range (if n > 64 then 1 else 0) 3)
    in
    let shape =
      Array.init rank (fun d ->
          if d = at then n else short.(if d < at then d else d - 1))
    in
    let* steps = layout in
    let* xs = array ~size:(constant (Ref.numel shape)) value in
    let t = lay_out steps (Nx.create dtype shape xs) in
    let nd = Nx.ndim t in
    let long = ref 0 in
    Array.iteri (fun d s -> if s > Nx.dim !long t then long := d) (Nx.shape t);
    let+ axis = frequency [ (3, constant !long); (1, int_range 0 (nd - 1)) ]
    and+ from_end = bool in
    (steps, t, if from_end then axis - nd else axis)
  in
  Gen.map
    (fun (_, t, axis) -> (t, axis))
    (Gen.with_pp
       (fun ppf (steps, t, axis) ->
         Format.fprintf ppf "axis %d of %a: %a" axis pp_layout steps (Ref.pp pp)
           (Ref.of_nx t))
       drawn)

let complex_signal = signal ~pp:pp_complex Nx.complex128 complex_values
let real_signal = signal ~pp:pp_float Nx.float64 values

let short_length =
  Gen.frequency
    [
      (4, Gen.int_range 0 5);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 7; 8; 11; 13; 17 ]);
    ]

(* A tensor of up to three short axes under a drawn layout, with distinct axes
   in a drawn order (none for the default), and sizes to crop or pad them to
   (none for their own). With [two] the axes are exactly two, or the default,
   and the tensor has at least two. *)
let block ?(two = false) ~pp dtype value =
  let drawn =
    let open Gen in
    let* shape =
      array ~size:(int_range (if two then 2 else 1) 3) short_length
    in
    let* steps = layout in
    let* xs = array ~size:(constant (Ref.numel shape)) value in
    let t = lay_out steps (Nx.create dtype shape xs) in
    let nd = Nx.ndim t in
    let all = List.init nd Fun.id in
    let* chosen =
      if two then
        frequency
          [
            (1, constant []);
            ( 3,
              map
                (fun l -> List.filteri (fun i _ -> i < 2) l)
                (permutation ~pp:Format.pp_print_int all) );
          ]
      else subsequence ~pp:Format.pp_print_int all
    in
    let* order = permutation ~pp:Format.pp_print_int chosen in
    let* from_end = array ~size:(constant (List.length order)) bool in
    let axes =
      if order = [] then None
      else
        Some (List.mapi (fun i a -> if from_end.(i) then a - nd else a) order)
    in
    let count =
      match axes with Some l -> List.length l | None -> if two then 2 else nd
    in
    let+ s = option (list ~size:(constant count) (int_range 1 8)) in
    (steps, t, axes, s)
  in
  Gen.map
    (fun (_, t, axes, s) -> (t, axes, s))
    (Gen.with_pp
       (fun ppf (steps, t, axes, s) ->
         Format.fprintf ppf "axes %a, s %a of %a: %a" (pp_option pp_ints) axes
           (pp_option pp_ints) s pp_layout steps (Ref.pp pp) (Ref.of_nx t))
       drawn)

let norm = Gen.option (Gen.of_list ~pp:pp_norm [ `Backward; `Forward; `Ortho ])

(* The axes a transform takes, [default] ones when [axes] is [None], made
   non-negative, and their output sizes. *)
let resolve (x : _ Ref.t) ~default axes s =
  let axes =
    List.map (Ref.axis x) (match axes with Some l -> l | None -> default)
  in
  let sizes =
    match s with Some s -> s | None -> List.map (fun a -> x.shape.(a)) axes
  in
  (axes, sizes)

let last_two nd = [ nd - 2; nd - 1 ]

(* The largest of each output size and the input's own, the points a bound
   counts. *)
let points (x : _ Ref.t) axes sizes =
  product (List.map2 (fun a n -> Int.max n x.shape.(a)) axes sizes)

(* Complex transforms *)

let complex_transforms =
  let transform name ~sign ~inverse nx =
    prop (name ^ " along an axis is the DFT, scaled as its norm says")
      (Gen.pair complex_signal norm) (fun ((t, axis), norm) ->
        let x = Ref.of_nx t in
        let n = x.shape.(Ref.axis x axis) in
        let s = factor ~inverse norm n in
        agrees complexes
          (Ref.map (scale s) (dft_along ~sign ~axes:[ axis ] ~sizes:[ n ] x))
          (bound ~scale:s ~axes:[ axis ] ~points:n cmag x)
          (Ref.of_nx (nx ~axis ?norm t)))
  in
  let resized name ~sign ~inverse nx =
    prop (name ^ " crops or zero-pads the axis to n before transforming")
      (Gen.pair complex_signal length) (fun ((t, axis), n) ->
        let x = Ref.of_nx t in
        let s = factor ~inverse None n in
        agrees complexes
          (Ref.map (scale s) (dft_along ~sign ~axes:[ axis ] ~sizes:[ n ] x))
          (bound ~scale:s ~axes:[ axis ]
             ~points:(Int.max n x.shape.(Ref.axis x axis))
             cmag x)
          (Ref.of_nx (nx ~axis ~n t)))
  in
  let parseval name ~inverse nx =
    prop
      (name
     ^ " multiplies the energy of each lane by n times its factor squared \
        (Parseval)") (Gen.pair complex_signal norm) (fun ((t, axis), norm) ->
        let x = Ref.of_nx t in
        let n = x.shape.(Ref.axis x axis) in
        let s = factor ~inverse norm n in
        let energy r =
          Ref.reduce ~axes:[ axis ] ~keepdims:true ( +. ) 0.
            (Ref.map
               (fun (z : Complex.t) -> (z.re *. z.re) +. (z.im *. z.im))
               r)
        in
        let magnitude =
          Ref.reduce ~axes:[ axis ] ~keepdims:true ( +. ) 0. (Ref.map cmag x)
        in
        let gain = if n = 0 then 0. else Float.of_int n *. s *. s in
        agrees reals
          (Ref.map (fun e -> gain *. e) (energy x))
          (Ref.map
             (fun m ->
               24. *. Float.of_int (n * (n + 1)) *. eps64 *. s *. s *. m *. m)
             magnitude)
          (energy (Ref.of_nx (nx ~axis ?norm t))))
  in
  group "fft and ifft"
    [
      transform "fft" ~sign:(-1) ~inverse:false (fun ~axis ?norm t ->
          Nx.fft ~axis ?norm t);
      transform "ifft" ~sign:1 ~inverse:true (fun ~axis ?norm t ->
          Nx.ifft ~axis ?norm t);
      resized "fft" ~sign:(-1) ~inverse:false (fun ~axis ~n t ->
          Nx.fft ~axis ~n t);
      resized "ifft" ~sign:1 ~inverse:true (fun ~axis ~n t ->
          Nx.ifft ~axis ~n t);
      prop "ifft inverts fft, and fft ifft, under each norm"
        (Gen.pair complex_signal norm) (fun ((t, axis), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          let w =
            tensor (cclose (worst (bound ~axes:[ axis ] ~points:n cmag x)))
          in
          Law.round_trip w w (Nx.fft ~axis ?norm) (Nx.ifft ~axis ?norm) t;
          Law.round_trip w w (Nx.ifft ~axis ?norm) (Nx.fft ~axis ?norm) t);
      prop "fft is linear: fft (a + k b) = fft a + k fft b"
        (Gen.bind complex_signal (fun (a, axis) ->
             Gen.map
               (fun (xs, k) ->
                 (a, Nx.create Nx.complex128 (Nx.shape a) xs, axis, k))
               (Gen.pair
                  (Gen.array ~size:(Gen.constant (Nx.numel a)) complex_values)
                  complex_values)))
        (fun (a, b, axis, k) ->
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          let n = ra.shape.(Ref.axis ra axis) in
          let combined =
            Ref.map2 (fun x y -> cmag x +. (cmag k *. cmag y)) ra rb
          in
          let w =
            tensor
              (cclose
                 (4. *. worst (bound ~axes:[ axis ] ~points:n Fun.id combined)))
          in
          let op x y = Nx.add x (Nx.mul_s y k) in
          Law.homomorphic w w (Nx.fft ~axis) op op (a, b));
      parseval "fft" ~inverse:false (fun ~axis ?norm t -> Nx.fft ~axis ?norm t);
      parseval "ifft" ~inverse:true (fun ~axis ?norm t -> Nx.ifft ~axis ?norm t);
      test "fft of one point is that point, and of an empty axis is empty"
        (fun () ->
          let one = Nx.create Nx.complex128 [| 1 |] [| c 5. (-3.) |] in
          equal (tensor (cclose 0.)) one (Nx.fft one);
          equal (tensor (cclose 0.)) one (Nx.ifft one);
          equal (array int) [| 2; 0 |]
            (Nx.shape (Nx.fft (Nx.zeros Nx.complex128 [| 2; 0 |]))));
    ]

(* Multi-axis complex transforms against the DFT along each axis. *)

let multi_axis_complex =
  let transform name ~sign ~inverse ~two ~default nx =
    prop
      (name
     ^ " is the DFT along each of its axes, cropped or zero-padded to s first")
      (Gen.pair (block ~two ~pp:pp_complex Nx.complex128 complex_values) norm)
      (fun ((t, axes, s), norm) ->
        let x = Ref.of_nx t in
        let axes', sizes = resolve x ~default:(default (Ref.ndim x)) axes s in
        let f = factor ~inverse norm (product sizes) in
        agrees complexes
          (Ref.map (scale f) (dft_along ~sign ~axes:axes' ~sizes x))
          (bound ~scale:f ~axes:axes' ~points:(points x axes' sizes) cmag x)
          (Ref.of_nx (nx ?axes ?s ?norm t)))
  in
  let all nd = List.init nd Fun.id in
  group "fft2 and fftn"
    [
      transform "fftn, over every axis by default," ~sign:(-1) ~inverse:false
        ~two:false ~default:all (fun ?axes ?s ?norm t ->
          Nx.fftn ?axes ?s ?norm t);
      transform "ifftn, over every axis by default," ~sign:1 ~inverse:true
        ~two:false ~default:all (fun ?axes ?s ?norm t ->
          Nx.ifftn ?axes ?s ?norm t);
      transform "fft2, over the last two axes by default," ~sign:(-1)
        ~inverse:false ~two:true ~default:last_two (fun ?axes ?s ?norm t ->
          Nx.fft2 ?axes ?s ?norm t);
      transform "ifft2, over the last two axes by default," ~sign:1
        ~inverse:true ~two:true ~default:last_two (fun ?axes ?s ?norm t ->
          Nx.ifft2 ?axes ?s ?norm t);
      prop "ifftn inverts fftn under each norm"
        (Gen.pair (block ~pp:pp_complex Nx.complex128 complex_values) norm)
        (fun ((t, axes, _), norm) ->
          let x = Ref.of_nx t in
          let axes', _ =
            resolve x ~default:(List.init (Ref.ndim x) Fun.id) axes None
          in
          let n = product (List.map (fun a -> x.shape.(a)) axes') in
          let w =
            tensor (cclose (worst (bound ~axes:axes' ~points:n cmag x)))
          in
          Law.round_trip w w (Nx.fftn ?axes ?norm) (Nx.ifftn ?axes ?norm) t);
    ]

(* Real transforms *)

(* [rfft] from one float dtype to one complex dtype, to the output's
   precision. *)
let rfft_to (type a b) name (idt : (float, a) Nx.dtype)
    (odt : (Complex.t, b) Nx.dtype) ~prec =
  prop
    ("rfft " ^ name
   ^ " along an axis is the first n/2 + 1 bins of the DFT, scaled as its norm \
      says")
    (Gen.pair (signal ~pp:pp_float idt values) norm)
    (fun ((t, axis), norm) ->
      let x = Ref.of_nx t in
      let n = x.shape.(Ref.axis x axis) in
      let s = factor ~inverse:false norm n in
      agrees complexes
        (Ref.map (scale s)
           (Ref.along ~axis ~length:((n / 2) + 1) (half_spectrum n) x))
        (bound ~prec ~scale:s ~axes:[ axis ] ~points:n Float.abs x)
        (Ref.of_nx (Nx.rfft odt ~axis ?norm t)))

(* [irfft] from one complex dtype to one float dtype, to the output's precision.
   The output length is drawn, or the default 2 (m - 1). *)
let irfft_to ?(value = complex_values) (type a b) name
    (idt : (Complex.t, a) Nx.dtype) (odt : (float, b) Nx.dtype) ~prec =
  prop
    ("irfft " ^ name
   ^ " along an axis is the inverse DFT of the Hermitian extension of its \
      bins, reading only the real part of DC and Nyquist")
    (Gen.triple (signal ~pp:pp_complex idt value) (Gen.option length) norm)
    (fun ((t, axis), drawn, norm) ->
      let x = Ref.of_nx t in
      let m = x.shape.(Ref.axis x axis) in
      assume (drawn <> None || m >= 1);
      let n = Option.value drawn ~default:(2 * (m - 1)) in
      let s = factor ~inverse:true norm n in
      agrees reals
        (Ref.along ~axis ~length:n
           (fun g ->
             Array.map (fun z -> s *. re z) (dft ~sign:1 (hermitian n g)))
           x)
        (bound ~prec ~scale:s ~axes:[ axis ] ~points:(Int.max n (2 * m)) cmag x)
        (Ref.of_nx (Nx.irfft odt ~axis ?n:drawn ?norm t)))

let real_transforms =
  group "rfft and irfft"
    [
      rfft_to "from float64 to complex128" Nx.float64 Nx.complex128 ~prec:f64;
      rfft_to "from float32 to complex128" Nx.float32 Nx.complex128 ~prec:f64;
      rfft_to "from float64 to complex64" Nx.float64 Nx.complex64 ~prec:f32;
      rfft_to "from float32 to complex64" Nx.float32 Nx.complex64 ~prec:f32;
      rfft_to "from float16 to complex64" Nx.float16 Nx.complex64 ~prec:f32;
      rfft_to "from bfloat16 to complex128" Nx.bfloat16 Nx.complex128 ~prec:f64;
      test "rfft and irfft read and write float8" (fun () ->
          let x = Nx.create Nx.float8_e4m3 [| 4 |] [| 1.; 0.5; -2.; 0.25 |] in
          let z = Nx.rfft Nx.complex64 x in
          equal
            (array (cclose 1e-6))
            [| c (-0.25) 0.; c 3. (-0.25); c (-1.75) 0. |]
            (Nx.to_array z);
          equal
            (array (rclose 0.))
            [| 1.; 0.5; -2.; 0.25 |]
            (Nx.to_array (Nx.irfft Nx.float8_e5m2 ~n:4 z)));
      prop "rfft crops or zero-pads the axis to n before transforming"
        (Gen.pair real_signal length) (fun ((t, axis), n) ->
          let x = Ref.of_nx t in
          agrees complexes
            (Ref.along ~axis ~length:((n / 2) + 1) (half_spectrum n) x)
            (bound ~axes:[ axis ]
               ~points:(Int.max n x.shape.(Ref.axis x axis))
               Float.abs x)
            (Ref.of_nx (Nx.rfft Nx.complex128 ~axis ~n t)));
      prop
        "rfft's bins are the conjugates of the bins fft gives at the mirrored \
         frequencies (Hermitian symmetry)"
        real_signal (fun (t, axis) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          let full = Ref.of_nx (Nx.fft ~axis (Nx.cast Nx.complex128 t)) in
          agrees complexes
            (Ref.along ~axis
               ~length:((n / 2) + 1)
               (fun l ->
                 Array.init
                   ((n / 2) + 1)
                   (fun k ->
                     if n = 0 then Complex.zero else conj l.((n - k) mod n)))
               full)
            (bound ~axes:[ axis ] ~points:n Float.abs x)
            (Ref.of_nx (Nx.rfft Nx.complex128 ~axis t)));
      test
        "at an even length, rfft's DC and Nyquist bins are exactly real \
         (nx.mli is silent)" (fun () ->
          for half = 1 to 70 do
            let n = 2 * half in
            let x =
              Nx.init Nx.float64 [| n |] (fun _ -> Random.float 2. -. 1.)
            in
            let z = Nx.to_array (Nx.rfft Nx.complex128 x) in
            equal ~msg:(Printf.sprintf "Im DC at %d" n) float_exact 0. z.(0).im;
            equal
              ~msg:(Printf.sprintf "Im Nyquist at %d" n)
              float_exact 0. z.(half).im
          done);
      irfft_to "from complex128 to float64" Nx.complex128 Nx.float64 ~prec:f64;
      irfft_to "from complex64 to float32" Nx.complex64 Nx.float32 ~prec:f32;
      irfft_to "from complex128 to float32" Nx.complex128 Nx.float32 ~prec:f32;
      irfft_to "from complex64 to float64" Nx.complex64 Nx.float64 ~prec:f64;
      (* Bins of magnitude one keep float16's sums below its largest float. *)
      irfft_to "from complex128 to float16" Nx.complex128 Nx.float16 ~prec:f16
        ~value:
          (Gen.with_pp pp_complex
             (Gen.map
                (fun (a, b) -> c a b)
                (Gen.pair (Gen.float_range (-1.) 1.) (Gen.float_range (-1.) 1.))));
      irfft_to "from complex64 to bfloat16" Nx.complex64 Nx.bfloat16 ~prec:bf16;
      prop
        "irfft with n inverts rfft with n, at odd lengths too, under each norm"
        (Gen.pair real_signal norm) (fun ((t, axis), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          let w =
            tensor (rclose (worst (bound ~axes:[ axis ] ~points:n Float.abs x)))
          in
          Law.round_trip w
            (tensor (cclose 0.))
            (Nx.rfft Nx.complex128 ~axis ~n ?norm)
            (Nx.irfft Nx.float64 ~axis ~n ?norm)
            t);
      prop
        "rfft keeps n times its factor squared of the energy, counting the \
         bins it drops (Parseval)" (Gen.pair real_signal norm)
        (fun ((t, axis), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          assume (n > 0);
          let s = factor ~inverse:false norm n in
          let lanes f r = Ref.reduce ~axes:[ axis ] ~keepdims:true f 0. r in
          let energy =
            Ref.along ~axis ~length:1
              (fun l ->
                let e = ref 0. in
                Array.iteri
                  (fun k (z : Complex.t) ->
                    let twice = k > 0 && 2 * k <> n in
                    e :=
                      !e
                      +. (if twice then 2. else 1.)
                         *. ((z.re *. z.re) +. (z.im *. z.im)))
                  l;
                [| !e |])
              (Ref.of_nx (Nx.rfft Nx.complex128 ~axis ?norm t))
          in
          agrees reals
            (Ref.map
               (fun e -> Float.of_int n *. s *. s *. e)
               (lanes (fun a v -> a +. (v *. v)) x))
            (Ref.map
               (fun m ->
                 48. *. Float.of_int (n * (n + 1)) *. eps64 *. s *. s *. m *. m)
               (lanes (fun a v -> a +. Float.abs v) x))
            energy);
      test
        "rfft normalises after storing, so a spectrum past complex64's range \
         stays infinite" (fun () ->
          let x = Nx.create Nx.float64 [| 2 |] [| 3e38; 3e38 |] in
          satisfies ~claim:"a component is infinite" (cclose 0.)
            (fun (z : Complex.t) ->
              Float.abs z.re = infinity || Float.abs z.im = infinity)
            (Nx.item [ 0 ] (Nx.rfft Nx.complex64 ~norm:`Forward x)));
      prop
        "fft and rfft carry a NaN or an infinity to every bin of its lane and \
         to no other lane"
        (Gen.pair
           (signal ~pp:pp_complex Nx.complex128
              (Gen.with_pp pp_complex
                 (Gen.map
                    (fun (a, b) -> c a b)
                    (Gen.pair non_finite non_finite))))
           (signal ~pp:pp_float Nx.float64 non_finite))
        (fun ((z, za), (x, xa)) ->
          let poisoned finite axis t =
            let p =
              Ref.reduce ~axes:[ axis ] ~keepdims:true ( || ) false
                (Ref.map (fun v -> not (finite v)) (Ref.of_nx t))
            in
            cover "a poisoned lane" (Array.exists Fun.id p.data);
            p
          in
          let check p y =
            equal (Ref.witness bool)
              (Ref.broadcast_to y.Ref.shape p)
              (Ref.map (fun v -> not (finite v)) y)
          in
          check (poisoned finite za z) (Ref.of_nx (Nx.fft ~axis:za z));
          check
            (poisoned Float.is_finite xa x)
            (Ref.of_nx (Nx.rfft Nx.complex128 ~axis:xa x)));
    ]

(* Multi-axis real transforms. rfftn transforms its last axis to n/2 + 1 bins
   and the others whole; the DFT along each axis commutes with the others, so
   the reference takes the last first. *)

let half_bins n lane =
  dft_at ~sign:(-1)
    (resize n Complex.zero lane)
    (Array.init ((n / 2) + 1) Fun.id)

let split_last l =
  let r = List.rev l in
  (List.rev (List.tl r), List.hd r)

let multi_axis_real =
  let all nd = List.init nd Fun.id in
  let forward name ~two ~default nx =
    prop
      (name
     ^ " is the DFT along each of its axes, cropped or zero-padded to s first, \
        keeping n/2 + 1 bins of the last")
      (Gen.pair (block ~two ~pp:pp_float Nx.float64 values) norm)
      (fun ((t, axes, s), norm) ->
        let x = Ref.of_nx t in
        let axes', sizes = resolve x ~default:(default (Ref.ndim x)) axes s in
        let leading, last = split_last axes' in
        let leading_sizes, n = split_last sizes in
        let f = factor ~inverse:false norm (product sizes) in
        let halved =
          Ref.along ~axis:last
            ~length:((n / 2) + 1)
            (half_bins n) (Ref.map of_real x)
        in
        agrees complexes
          (Ref.map (scale f)
             (dft_along ~sign:(-1) ~axes:leading ~sizes:leading_sizes halved))
          (bound ~scale:f ~axes:axes' ~points:(points x axes' sizes) Float.abs x)
          (Ref.of_nx (nx ?axes ?s ?norm t)))
  in
  let inverse name ~two ~default nx =
    prop
      (name
     ^ " is the inverse DFT along its leading axes, then irfft along its last, \
        each cropped or zero-padded to s, the last to 2 (m - 1) by default")
      (Gen.pair (block ~two ~pp:pp_complex Nx.complex128 complex_values) norm)
      (fun ((t, axes, s), norm) ->
        let x = Ref.of_nx t in
        let axes', own = resolve x ~default:(default (Ref.ndim x)) axes None in
        let leading, last = split_last axes' in
        let m = x.shape.(last) in
        assume (s <> None || m >= 1);
        let sizes =
          match s with
          | Some s -> s
          | None -> fst (split_last own) @ [ 2 * (m - 1) ]
        in
        let leading_sizes, n = split_last sizes in
        let f = factor ~inverse:true norm (product sizes) in
        let expected =
          Ref.along ~axis:last ~length:n
            (fun g ->
              Array.map (fun z -> f *. re z) (dft ~sign:1 (hermitian n g)))
            (dft_along ~sign:1 ~axes:leading ~sizes:leading_sizes x)
        in
        agrees reals expected
          (bound ~scale:f ~axes:axes' ~points:(2 * points x axes' sizes) cmag x)
          (Ref.of_nx (nx ?axes ?s ?norm t)))
  in
  group "rfft2 and rfftn"
    [
      forward "rfftn, over every axis by default," ~two:false ~default:all
        (fun ?axes ?s ?norm t -> Nx.rfftn Nx.complex128 ?axes ?s ?norm t);
      forward "rfft2, over the last two axes by default," ~two:true
        ~default:last_two (fun ?axes ?s ?norm t ->
          Nx.rfft2 Nx.complex128 ?axes ?s ?norm t);
      inverse "irfftn, over every axis by default," ~two:false ~default:all
        (fun ?axes ?s ?norm t -> Nx.irfftn Nx.float64 ?axes ?s ?norm t);
      inverse "irfft2, over the last two axes by default," ~two:true
        ~default:last_two (fun ?axes ?s ?norm t ->
          Nx.irfft2 Nx.float64 ?axes ?s ?norm t);
      prop "irfftn with s inverts rfftn, at odd lengths too"
        (Gen.pair (block ~pp:pp_float Nx.float64 values) norm)
        (fun ((t, axes, _), norm) ->
          let x = Ref.of_nx t in
          let axes', sizes = resolve x ~default:(all (Ref.ndim x)) axes None in
          let w =
            tensor
              (rclose
                 (worst
                    (bound ~axes:axes' ~points:(2 * product sizes) Float.abs x)))
          in
          Law.round_trip w
            (tensor (cclose 0.))
            (Nx.rfftn Nx.complex128 ?axes ?norm)
            (Nx.irfftn Nx.float64 ?axes ~s:sizes ?norm)
            t);
      prop
        "rfftn to complex64 rounds the spectrum it carries between axes to \
         float32 precision" (block ~pp:pp_float Nx.float32 values)
        (fun (t, axes, _) ->
          let x = Ref.of_nx t in
          let axes', sizes = resolve x ~default:(all (Ref.ndim x)) axes None in
          let leading, last = split_last axes' in
          let leading_sizes, n = split_last sizes in
          agrees complexes
            (dft_along ~sign:(-1) ~axes:leading ~sizes:leading_sizes
               (Ref.along ~axis:last
                  ~length:((n / 2) + 1)
                  (half_bins n) (Ref.map of_real x)))
            (bound ~prec:f32 ~axes:axes' ~points:(product sizes) Float.abs x)
            (Ref.of_nx (Nx.rfftn Nx.complex64 ?axes t)));
    ]

(* Hermitian transforms: hfft is the forward transform of the Hermitian signal
   whose half an irfft reads, ihfft its inverse. *)

let hermitian_transforms =
  group "hfft and ihfft"
    [
      prop
        "hfft along an axis is the DFT of the Hermitian signal its bins \
         describe, n = 2 (m - 1) points by default, scaled as its norm says"
        (Gen.triple complex_signal (Gen.option length) norm)
        (fun ((t, axis), drawn, norm) ->
          let x = Ref.of_nx t in
          let m = x.shape.(Ref.axis x axis) in
          assume (drawn <> None || m >= 1);
          let n = Option.value drawn ~default:(2 * (m - 1)) in
          let s = factor ~inverse:false norm n in
          agrees reals
            (Ref.along ~axis ~length:n
               (fun g ->
                 Array.map (fun z -> s *. re z) (dft ~sign:(-1) (hermitian n g)))
               x)
            (bound ~scale:s ~axes:[ axis ] ~points:(Int.max n (2 * m)) cmag x)
            (Ref.of_nx (Nx.hfft Nx.float64 ~axis ?n:drawn ?norm t)));
      prop
        "ihfft along an axis is the first n/2 + 1 bins of the inverse DFT, \
         scaled as its norm says" (Gen.pair real_signal norm)
        (fun ((t, axis), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          let s = factor ~inverse:true norm n in
          agrees complexes
            (Ref.along ~axis
               ~length:((n / 2) + 1)
               (fun l ->
                 Array.map (scale s)
                   (dft_at ~sign:1 (Array.map of_real l)
                      (Array.init ((n / 2) + 1) Fun.id)))
               x)
            (bound ~scale:s ~axes:[ axis ] ~points:n Float.abs x)
            (Ref.of_nx (Nx.ihfft Nx.complex128 ~axis ?norm t)));
      prop "hfft with n inverts ihfft with n, under each norm"
        (Gen.pair real_signal norm) (fun ((t, axis), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          let w =
            tensor (rclose (worst (bound ~axes:[ axis ] ~points:n Float.abs x)))
          in
          Law.round_trip w
            (tensor (cclose 0.))
            (Nx.ihfft Nx.complex128 ~axis ~n ?norm)
            (Nx.hfft Nx.float64 ~axis ~n ?norm)
            t);
    ]

(* Cosine and sine transforms against their sums, with the factor of two of the
   DFT of the symmetric extension they are (nx.mli names only its logical
   length) and the usual orthonormal boundary weights. *)

let cosine_sum type_ (x : float array) =
  let n = Array.length x in
  let term j num den =
    2. *. x.(j) *. Float.cos (Float.pi *. Float.of_int num /. Float.of_int den)
  in
  Array.init n (fun k ->
      let s = ref 0. in
      (match type_ with
      | 1 ->
          s := x.(0) +. ((if k mod 2 = 0 then 1. else -1.) *. x.(n - 1));
          for j = 1 to n - 2 do
            s := !s +. term j (k * j) (n - 1)
          done
      | 2 ->
          for j = 0 to n - 1 do
            s := !s +. term j (k * ((2 * j) + 1)) (2 * n)
          done
      | 3 ->
          s := x.(0);
          for j = 1 to n - 1 do
            s := !s +. term j (((2 * k) + 1) * j) (2 * n)
          done
      | _ ->
          for j = 0 to n - 1 do
            s := !s +. term j (((2 * k) + 1) * ((2 * j) + 1)) (4 * n)
          done);
      !s)

let sine_sum type_ (x : float array) =
  let n = Array.length x in
  let term j num den =
    2. *. x.(j) *. Float.sin (Float.pi *. Float.of_int num /. Float.of_int den)
  in
  Array.init n (fun k ->
      let s = ref 0. in
      (match type_ with
      | 1 ->
          for j = 0 to n - 1 do
            s := !s +. term j ((k + 1) * (j + 1)) (n + 1)
          done
      | 2 ->
          for j = 0 to n - 1 do
            s := !s +. term j ((k + 1) * ((2 * j) + 1)) (2 * n)
          done
      | 3 ->
          s := (if k mod 2 = 0 then 1. else -1.) *. x.(n - 1);
          for j = 0 to n - 2 do
            s := !s +. term j (((2 * k) + 1) * (j + 1)) (2 * n)
          done
      | _ ->
          for j = 0 to n - 1 do
            s := !s +. term j (((2 * k) + 1) * ((2 * j) + 1)) (4 * n)
          done);
      !s)

let logical_length family type_ n =
  match (family, type_) with
  | `Dct, 1 -> 2 * (n - 1)
  | `Dst, 1 -> 2 * (n + 1)
  | _ -> 2 * n

(* The transform of [x]. An inverse is the transform of the inverse type, II and
   III swapping, with the norm's factor swapped. *)
let trig_ref ~family ~inverse ~type_ ~norm x =
  let n = Array.length x in
  let raw =
    if inverse then match type_ with 2 -> 3 | 3 -> 2 | t -> t else type_
  in
  let ortho = norm = Some `Ortho and r2 = Float.sqrt 2. in
  let x = Array.copy x in
  (if ortho then
     match (family, raw) with
     | `Dct, 1 ->
         x.(0) <- x.(0) *. r2;
         x.(n - 1) <- x.(n - 1) *. r2
     | `Dct, 3 -> x.(0) <- x.(0) *. r2
     | `Dst, 3 -> x.(n - 1) <- x.(n - 1) *. r2
     | _ -> ());
  let y = (match family with `Dct -> cosine_sum | `Dst -> sine_sum) raw x in
  (if ortho then
     match (family, raw) with
     | `Dct, 1 ->
         y.(0) <- y.(0) /. r2;
         y.(n - 1) <- y.(n - 1) /. r2
     | `Dct, 2 -> y.(0) <- y.(0) /. r2
     | `Dst, 2 -> y.(n - 1) <- y.(n - 1) /. r2
     | _ -> ());
  let s = factor ~inverse norm (logical_length family type_ n) in
  Array.map (fun v -> s *. v) y

let pp_family ppf f =
  Format.pp_print_string ppf (match f with `Dct -> "dct" | `Dst -> "dst")

let trig ~family ~inverse ~type_ ?axis ?norm t =
  match (family, inverse) with
  | `Dct, false -> Nx.dct ~type_ ?axis ?norm t
  | `Dct, true -> Nx.idct ~type_ ?axis ?norm t
  | `Dst, false -> Nx.dst ~type_ ?axis ?norm t
  | `Dst, true -> Nx.idst ~type_ ?axis ?norm t

let trig_n ~family ~inverse ~type_ ?axes ?norm t =
  match (family, inverse) with
  | `Dct, false -> Nx.dctn ~type_ ?axes ?norm t
  | `Dct, true -> Nx.idctn ~type_ ?axes ?norm t
  | `Dst, false -> Nx.dstn ~type_ ?axes ?norm t
  | `Dst, true -> Nx.idstn ~type_ ?axes ?norm t

(* Whether a transform refuses an axis of length [n]. *)
let refused family type_ n = n = 0 || (family = `Dct && type_ = 1 && n = 1)

let trig_length =
  Gen.frequency
    [
      (4, Gen.int_range 1 12);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 16; 17; 31; 32; 64 ]);
    ]

let kind =
  Gen.triple
    (Gen.of_list ~pp:pp_family [ `Dct; `Dst ])
    Gen.bool (Gen.int_range 1 4)

let trig_signal dtype = signal ~length:trig_length ~pp:pp_float dtype values

let trig_transforms =
  group "dct and dst"
    [
      prop
        "dct, dst, idct and idst of each type are their sums of cosines and \
         sines, scaled as the norm says, and refuse an empty axis and a \
         one-point DCT-I"
        (Gen.triple (trig_signal Nx.float64) kind norm)
        (fun ((t, axis), (family, inverse, type_), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          let run () = trig ~family ~inverse ~type_ ~axis ?norm t in
          if refused family type_ n then raises_invalid_arg run
          else
            agrees reals
              (Ref.along ~axis ~length:n
                 (trig_ref ~family ~inverse ~type_ ~norm)
                 x)
              (bound ~scale:4. ~axes:[ axis ]
                 ~points:(logical_length family type_ n)
                 Float.abs x)
              (Ref.of_nx (run ())));
      prop "dct and dst of float32 are float32, to float32 precision"
        (Gen.pair (trig_signal Nx.float32) kind)
        (fun ((t, axis), (family, inverse, type_)) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          assume (not (refused family type_ n));
          agrees reals
            (Ref.along ~axis ~length:n
               (trig_ref ~family ~inverse ~type_ ~norm:None)
               x)
            (bound ~prec:f32 ~scale:4. ~axes:[ axis ]
               ~points:(logical_length family type_ n)
               Float.abs x)
            (Ref.of_nx (trig ~family ~inverse ~type_ ~axis t)));
      prop "idct inverts dct, and idst dst, of each type under each norm"
        (Gen.triple (trig_signal Nx.float64) kind norm)
        (fun ((t, axis), (family, _, type_), norm) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          assume (not (refused family type_ n));
          let w =
            tensor
              (rclose
                 (worst
                    (bound ~scale:16. ~axes:[ axis ]
                       ~points:(logical_length family type_ n)
                       Float.abs x)))
          in
          Law.round_trip w w
            (trig ~family ~inverse:false ~type_ ~axis ?norm)
            (trig ~family ~inverse:true ~type_ ~axis ?norm)
            t);
      prop "under `Ortho, dct and dst keep the energy of each lane"
        (Gen.pair (trig_signal Nx.float64) kind)
        (fun ((t, axis), (family, inverse, type_)) ->
          let x = Ref.of_nx t in
          let n = x.shape.(Ref.axis x axis) in
          assume (not (refused family type_ n));
          let lanes f r = Ref.reduce ~axes:[ axis ] ~keepdims:true f 0. r in
          let energy = lanes (fun a v -> a +. (v *. v)) in
          agrees reals (energy x)
            (Ref.map
               (fun m ->
                 64. *. Float.of_int ((n + 1) * (n + 1)) *. eps64 *. m *. m)
               (lanes (fun a v -> a +. Float.abs v) x))
            (energy
               (Ref.of_nx (trig ~family ~inverse ~type_ ~axis ~norm:`Ortho t))));
      prop
        "dctn, dstn and their inverses apply the transform along each of their \
         axes, every one by default"
        (Gen.triple (block ~pp:pp_float Nx.float64 values) kind norm)
        (fun ((t, axes, _), (family, inverse, type_), norm) ->
          let x = Ref.of_nx t in
          let axes', _ =
            resolve x ~default:(List.init (Ref.ndim x) Fun.id) axes None
          in
          let run () = trig_n ~family ~inverse ~type_ ?axes ?norm t in
          if List.exists (fun a -> refused family type_ x.shape.(a)) axes' then
            raises_invalid_arg run
          else
            let expected =
              List.fold_left
                (fun r a ->
                  Ref.along ~axis:a ~length:r.Ref.shape.(a)
                    (trig_ref ~family ~inverse ~type_ ~norm)
                    r)
                x axes'
            in
            let points =
              product
                (List.map
                   (fun a -> logical_length family type_ x.shape.(a))
                   axes')
            in
            agrees reals expected
              (bound
                 ~scale:(4. ** Float.of_int (List.length axes'))
                 ~axes:axes' ~points Float.abs x)
              (Ref.of_nx (run ())));
      test "dctn and dstn over no axes, or of a scalar, return their input"
        (fun () ->
          let x =
            Nx.create Nx.float64 [| 2; 3 |] [| 1.; -2.; 0.5; 4.; 3.; -1.5 |]
          in
          let exact = tensor float_exact in
          equal exact x (Nx.dctn ~axes:[] x);
          equal exact x (Nx.idstn ~axes:[] x);
          let s = Nx.scalar Nx.float64 2. in
          equal exact s (Nx.dctn s);
          equal exact s (Nx.dstn s));
    ]

(* Frequencies and shifts *)

let pp_int32 ppf v = Format.fprintf ppf "%ld" v
let spacing = Gen.of_list ~pp:pp_float [ 1.; 0.5; 2.; 0.1; 3.; 1e-3; 44100. ]

let frequencies =
  let check ~eps expected actual =
    equal
      (Ref.witness (close ~rel:(2. *. eps) ()))
      (Ref.create [| Array.length expected |] expected)
      (Ref.of_nx actual)
  in
  group "fftfreq and rfftfreq"
    [
      prop
        "fftfreq counts 0, 1, ... up to the middle, then the negative \
         frequencies, over d n"
        (Gen.pair (Gen.int_range 1 64) spacing)
        (fun (n, d) ->
          let f i =
            Float.of_int (if i < (n + 1) / 2 then i else i - n)
            /. (d *. Float.of_int n)
          in
          let expected = Array.init n f in
          check ~eps:eps64 expected (Nx.fftfreq Nx.float64 ~d n);
          check ~eps:f32.eps expected (Nx.fftfreq Nx.float32 ~d n));
      prop "rfftfreq counts 0, 1, ..., n/2 over d n, the bins of rfft"
        (Gen.pair (Gen.int_range 1 64) spacing)
        (fun (n, d) ->
          let expected =
            Array.init
              ((n / 2) + 1)
              (fun i -> Float.of_int i /. (d *. Float.of_int n))
          in
          check ~eps:eps64 expected (Nx.rfftfreq Nx.float64 ~d n);
          check ~eps:f32.eps expected (Nx.rfftfreq Nx.float32 ~d n));
      test "fftfreq of 4 and 1 point spacing is [0; 0.25; -0.5; -0.25]"
        (fun () ->
          equal (array float_exact)
            [| 0.; 0.25; -0.5; -0.25 |]
            (Nx.to_array (Nx.fftfreq Nx.float64 4)));
    ]

let shifts =
  let drawn =
    let open Gen in
    let* t =
      viewed ~pp:pp_int32 Nx.int32 (map Int32.of_int (int_range (-9) 9))
    in
    let nd = Nx.ndim t in
    let* chosen = subsequence ~pp:Format.pp_print_int (List.init nd Fun.id) in
    let* from_end = array ~size:(constant (List.length chosen)) bool in
    let+ every = bool in
    let axes =
      List.mapi (fun i a -> if from_end.(i) then a - nd else a) chosen
    in
    (t, if every then None else Some axes)
  in
  let rolled sign (t, axes) =
    let x = Ref.of_nx t in
    let axes =
      match axes with None -> List.init (Ref.ndim x) Fun.id | Some l -> l
    in
    List.fold_left
      (fun r a -> Ref.roll ~axis:a (sign * (r.Ref.shape.(Ref.axis r a) / 2)) r)
      x axes
  in
  let exact = tensor int32 in
  group "fftshift and ifftshift"
    [
      prop
        "fftshift rolls each of its axes, every one by default, forward by \
         half its length, and ifftshift back"
        drawn (fun (t, axes) ->
          equal (Ref.witness int32)
            (rolled 1 (t, axes))
            (Ref.of_nx (Nx.fftshift ?axes t));
          equal (Ref.witness int32)
            (rolled (-1) (t, axes))
            (Ref.of_nx (Nx.ifftshift ?axes t)));
      prop "ifftshift undoes fftshift, and fftshift ifftshift" drawn
        (fun (t, axes) ->
          Law.round_trip exact exact (Nx.fftshift ?axes) (Nx.ifftshift ?axes) t;
          Law.round_trip exact exact (Nx.ifftshift ?axes) (Nx.fftshift ?axes) t);
      prop "fftshift puts fftfreq's frequencies in ascending order"
        (Gen.int_range 1 64) (fun n ->
          equal
            (Ref.witness (close ~rel:(2. *. eps64) ()))
            (Ref.init [| n |] (fun i ->
                 Float.of_int (i.(0) - (n / 2)) /. Float.of_int n))
            (Ref.of_nx (Nx.fftshift (Nx.fftfreq Nx.float64 n))));
    ]

(* Short-time analysis *)

let hann_ref n =
  Array.init n (fun i ->
      0.5
      -. (0.5 *. Float.cos (2. *. Float.pi *. Float.of_int i /. Float.of_int n)))

let default_step window = Int.max 1 (window / 4)

(* The frames of [lane] of [window] samples every [step], each times [w]. *)
let frames ~window ~step w lane =
  let count = ((Array.length lane - window) / step) + 1 in
  let frame f = Array.init window (fun j -> lane.((f * step) + j) *. w.(j)) in
  List.init count frame

(* The first window/2 + 1 bins of the DFT of each frame, frame after frame, and
   each bin's bound. *)
let stft_ref ~window ~step w (x : float Ref.t) =
  let bins = (window / 2) + 1 in
  let count = ((x.shape.(Ref.ndim x - 1) - window) / step) + 1 in
  let per_frame f =
    Ref.along ~axis:(-1) ~length:(count * bins)
      (fun lane -> Array.concat (List.map f (frames ~window ~step w lane)))
      x
    |> Ref.reshape
         (Array.append (Array.sub x.shape 0 (Ref.ndim x - 1)) [| count; bins |])
  in
  ( per_frame (half_spectrum window),
    per_frame (fun fr ->
        Array.make bins
          (8.
          *. Float.of_int (window + 1)
          *. ((eps64 *. Array.fold_left (fun a v -> a +. Float.abs v) 0. fr)
             +. f64.tiny))) )

let framing =
  let drawn =
    let open Gen in
    let* batch = array ~size:(int_range 0 2) (int_range 0 2) in
    let* n = int_range 1 48 in
    let shape = Array.append batch [| n |] in
    let* xs = array ~size:(constant (Ref.numel shape)) values in
    let* steps = layout in
    let t = lay_out steps (Nx.create Nx.float64 shape xs) in
    let* window = int_range 1 (Int.max 1 (Nx.dim (-1) t)) in
    let* step = option (int_range 1 (window + 3)) in
    let+ win = option (array ~size:(constant window) values) in
    (steps, t, window, step, win)
  in
  Gen.map
    (fun (_, t, window, step, win) -> (t, window, step, win))
    (Gen.with_pp
       (fun ppf (steps, t, window, step, win) ->
         Format.fprintf ppf "window %d, step %a, win %a, of %a: %a" window
           (pp_option Format.pp_print_int)
           step
           (pp_option (fun ppf a ->
                Ref.pp pp_float ppf (Ref.create [| Array.length a |] a)))
           win pp_layout steps (Ref.pp pp_float) (Ref.of_nx t))
       drawn)

(* A signal, a window, a step that covers it, a taper (the default, or drawn
   with zeros, whose samples no frame recovers) and an output length. *)
let reconstruction =
  let weight =
    Gen.frequency
      [
        (1, Gen.constant ~pp:pp_float 0.);
        (4, Gen.float_range 0.25 1.);
        (2, Gen.float_range (-1.) (-0.25));
      ]
  in
  let drawn =
    let open Gen in
    let* batch = array ~size:(int_range 0 2) (int_range 1 2) in
    let* n = int_range 1 48 in
    let shape = Array.append batch [| n |] in
    let* xs = array ~size:(constant (Ref.numel shape)) values in
    let* window = int_range 1 n in
    let* step = option (int_range 1 window) in
    let* win = option (array ~size:(constant window) weight) in
    let+ length = option (int_range 1 (n + 4)) in
    (Nx.create Nx.float64 shape xs, window, step, win, length)
  in
  Gen.with_pp
    (fun ppf (t, window, step, win, length) ->
      let opt = pp_option Format.pp_print_int in
      Format.fprintf ppf "window %d, step %a, win %a, length %a, of %a" window
        opt step
        (pp_option (fun ppf a ->
             Ref.pp pp_float ppf (Ref.create [| Array.length a |] a)))
        win opt length (Ref.pp pp_float) (Ref.of_nx t))
    drawn

let short_time =
  let window_of a = Nx.create Nx.float64 [| Array.length a |] a in
  group "hann, stft and istft"
    [
      cases "hann is 0.5 - 0.5 cos (2 pi i / n), the periodic taper"
        ~name:string_of_int [ 1; 2; 3; 4; 5; 8; 16; 17; 64 ] (fun n ->
          equal
            (Ref.witness (close ~abs:(4. *. eps64) ~rel:0. ()))
            (Ref.create [| n |] (hann_ref n))
            (Ref.of_nx (Nx.hann Nx.float64 n)));
      test "hann of 4 is [0; 0.5; 1; 0.5]" (fun () ->
          equal
            (array (close ~abs:eps64 ~rel:0. ()))
            [| 0.; 0.5; 1.; 0.5 |]
            (Nx.to_array (Nx.hann Nx.float64 4)));
      prop "k copies of hann shifted by n / k sum to k / 2"
        (Gen.pair (Gen.int_range 2 6) (Gen.int_range 1 10))
        (fun (k, hop) ->
          let n = k * hop in
          let w = Nx.hann Nx.float64 n in
          let sum =
            List.fold_left
              (fun s j -> Nx.add s (Nx.roll (j * hop) w))
              (Nx.zeros Nx.float64 [| n |])
              (List.init k Fun.id)
          in
          equal
            (tensor (close ~abs:(8. *. Float.of_int k *. eps64) ~rel:0. ()))
            (Nx.full Nx.float64 [| n |] (Float.of_int k /. 2.))
            sum);
      prop
        "stft is the DFT of each frame of window samples every step (window / \
         4, at least 1, by default) times win (a periodic Hann by default), \
         and refuses a window past the last axis"
        framing (fun (t, window, step, win) ->
          let x = Ref.of_nx t in
          let run () =
            Nx.stft Nx.complex128 ~window ?step ?win:(Option.map window_of win)
              t
          in
          if window > x.shape.(Ref.ndim x - 1) then raises_invalid_arg run
          else
            let w = Option.value win ~default:(hann_ref window) in
            let step = Option.value step ~default:(default_step window) in
            let expected, bound = stft_ref ~window ~step w x in
            agrees complexes expected bound (Ref.of_nx (run ())));
      cases "stft cuts (n - window) / step + 1 frames of window / 2 + 1 bins"
        ~name:(fun (n, window, step, _) ->
          Printf.sprintf "%d samples, window %d, step %s" n window
            (match step with Some s -> string_of_int s | None -> "default"))
        [
          (64, 16, Some 8, [| 7; 9 |]);
          (64, 16, None, [| 13; 9 |]);
          (64, 2, None, [| 63; 2 |]);
          (9, 4, None, [| 6; 3 |]);
          (16, 16, Some 5, [| 1; 9 |]);
          (20, 16, Some 5, [| 1; 9 |]);
          (21, 16, Some 5, [| 2; 9 |]);
          (5, 1, Some 1, [| 5; 1 |]);
          (7, 3, Some 10, [| 1; 2 |]);
          (10, 3, Some 3, [| 3; 2 |]);
        ]
        (fun (n, window, step, shape) ->
          equal (array int) shape
            (Nx.shape
               (Nx.stft Nx.complex128 ~window ?step
                  (Nx.zeros Nx.float64 [| n |]))));
      cases "istft gives (frames - 1) step + window samples"
        ~name:(fun (frames, window, step, _) ->
          Printf.sprintf "%d frames, window %d, step %s" frames window
            (match step with Some s -> string_of_int s | None -> "default"))
        [
          (7, 16, Some 8, 64);
          (13, 16, None, 64);
          (1, 5, Some 2, 5);
          (3, 3, Some 1, 5);
          (4, 2, None, 5);
        ]
        (fun (frames, window, step, samples) ->
          equal (array int) [| samples |]
            (Nx.shape
               (Nx.istft Nx.float64 ~window ?step
                  (Nx.zeros Nx.complex128 [| frames; (window / 2) + 1 |]))));
      prop
        "istft inverts stft wherever a frame's taper reaches, gives 0 where \
         none does, and crops or zero-extends to length"
        reconstruction (fun (t, window, step, win, length) ->
          let w = Option.value win ~default:(hann_ref window) in
          let hop = Option.value step ~default:(default_step window) in
          let n = Nx.dim (-1) t in
          let count = ((n - window) / hop) + 1 in
          let out = ((count - 1) * hop) + window in
          let final = Option.value length ~default:out in
          let reach p =
            List.fold_left
              (fun (env, taps) f ->
                let j = p - (f * hop) in
                if j >= 0 && j < window then
                  (env +. (w.(j) *. w.(j)), taps +. Float.abs w.(j))
                else (env, taps))
              (0., 0.) (List.init count Fun.id)
          in
          let peak =
            Array.fold_left (fun a v -> Float.max a (Float.abs v)) 0. w
          in
          let lane f =
            Ref.along ~axis:(-1) ~length:final (fun l ->
                Array.init final (fun p ->
                    if p >= out then 0.
                    else
                      let env, taps = reach p in
                      if env = 0. then 0. else f l p env taps))
          in
          let x = Ref.of_nx t in
          cover "a sample no frame recovers"
            (List.exists (fun p -> fst (reach p) = 0.) (List.init out Fun.id));
          cover "zero-extended" (final > out);
          cover "cropped" (final < out);
          let win = Option.map window_of win in
          agrees reals
            (lane (fun l p _ _ -> l.(p)) x)
            (lane
               (fun l _ env taps ->
                 16.
                 *. Float.of_int (window + 1)
                 *. (eps64 *. peak *. taps /. env
                     *. Array.fold_left (fun a v -> a +. Float.abs v) 0. l
                    +. f64.tiny))
               x)
            (Ref.of_nx
               (Nx.istft Nx.float64 ~window ?step ?win ?length
                  (Nx.stft Nx.complex128 ~window ?step ?win t))));
    ]

(* Lengths *)

let random_complex () = c (Random.float 2. -. 1.) (Random.float 2. -. 1.)

let lengths =
  let sum_mag f a = Array.fold_left (fun s v -> s +. f v) 0. a in
  let tol n m = 8. *. Float.of_int (n + 1) *. eps64 *. m in
  group "lengths"
    [
      test "fft, rfft and irfft are the DFT at every length up to 256"
        (fun () ->
          for n = 1 to 256 do
            let msg = Printf.sprintf "length %d" n in
            let z = Array.init n (fun _ -> random_complex ()) in
            let x = Array.init n (fun _ -> Random.float 2. -. 1.) in
            let g = Array.init ((n / 2) + 1) (fun _ -> random_complex ()) in
            equal ~msg
              (array (cclose (tol n (sum_mag cmag z))))
              (dft ~sign:(-1) z)
              (Nx.to_array (Nx.fft (Nx.create Nx.complex128 [| n |] z)));
            equal ~msg
              (array (cclose (tol n (sum_mag Float.abs x))))
              (half_spectrum n x)
              (Nx.to_array
                 (Nx.rfft Nx.complex128 (Nx.create Nx.float64 [| n |] x)));
            equal ~msg
              (array (rclose (tol (2 * n) (sum_mag cmag g))))
              (Array.map re (dft ~sign:1 (hermitian n g)))
              (Nx.to_array
                 (Nx.irfft Nx.float64 ~n ~norm:`Forward
                    (Nx.create Nx.complex128 [| Array.length g |] g)))
          done);
      cases "long signals are the DFT at sampled bins, and round-trip"
        ~name:string_of_int [ 4096; 4099; 8192; 44100; 65535; 65536; 131042 ]
        (fun n ->
          let ks =
            List.sort_uniq compare
              [ 0; 1; 2; 3; n / 3; (n / 2) - 1; n / 2; (n / 2) + 1; n - 1 ]
            |> Array.of_list
          in
          let z = Array.init n (fun _ -> random_complex ()) in
          let x = Array.init n (fun _ -> Random.float 2. -. 1.) in
          let tz = Nx.create Nx.complex128 [| n |] z in
          let tx = Nx.create Nx.float64 [| n |] x in
          let cz = cclose (tol n (sum_mag cmag z)) in
          let cx = cclose (tol n (sum_mag Float.abs x)) in
          let at t ks = Array.map (fun k -> Nx.item [ k ] t) ks in
          equal ~msg:"fft" (array cz) (dft_at ~sign:(-1) z ks)
            (at (Nx.fft tz) ks);
          equal ~msg:"ifft" (array cz) (dft_at ~sign:1 z ks)
            (at (Nx.ifft ~norm:`Forward tz) ks);
          let half =
            Array.of_list (List.filter (fun k -> k <= n / 2) (Array.to_list ks))
          in
          equal ~msg:"rfft" (array cx)
            (dft_at ~sign:(-1) (Array.map of_real x) half)
            (at (Nx.rfft Nx.complex128 tx) half);
          equal ~msg:"ifft of fft" (array cz) z
            (Nx.to_array (Nx.ifft (Nx.fft tz)));
          equal ~msg:"irfft of rfft"
            (array (rclose (tol n (sum_mag Float.abs x))))
            x
            (Nx.to_array (Nx.irfft Nx.float64 ~n (Nx.rfft Nx.complex128 tx))));
      test "a batch of long lines transforms each line as it would alone"
        (fun () ->
          let lines = 64 and n = 4096 in
          let x =
            Nx.init Nx.float64 [| lines; n |] (fun _ -> Random.float 2. -. 1.)
          in
          let spectra = Nx.rfft Nx.complex128 x in
          let ks = [| 0; 1; 5; 1000; 2047; 2048 |] in
          List.iter
            (fun l ->
              let line = Array.init n (fun j -> Nx.item [ l; j ] x) in
              equal
                ~msg:(Printf.sprintf "rfft line %d" l)
                (array (cclose (tol n (sum_mag Float.abs line))))
                (dft_at ~sign:(-1) (Array.map of_real line) ks)
                (Array.map (fun k -> Nx.item [ l; k ] spectra) ks))
            [ 0; 17; 34; 51; 63 ];
          let lines = 16 and n = 8192 in
          let g =
            Nx.init Nx.complex128
              [| lines; (n / 2) + 1 |]
              (fun _ -> random_complex ())
          in
          let signals = Nx.irfft Nx.float64 ~n ~norm:`Forward g in
          let js = [| 0; 1; 7; 4095; 4096; 8191 |] in
          List.iter
            (fun l ->
              let bins =
                Array.init ((n / 2) + 1) (fun k -> Nx.item [ l; k ] g)
              in
              equal
                ~msg:(Printf.sprintf "irfft line %d" l)
                (array (rclose (tol (2 * n) (sum_mag cmag bins))))
                (Array.map re (dft_at ~sign:1 (hermitian n bins) js))
                (Array.map (fun j -> Nx.item [ l; j ] signals) js))
            [ 0; 5; 10; 15 ]);
    ]

(* Errors *)

let errors =
  let v = Nx.ones Nx.complex128 [| 4 |] and r = Nx.ones Nx.float64 [| 4 |] in
  let m = Nx.ones Nx.complex128 [| 2; 4 |]
  and rm = Nx.ones Nx.float64 [| 2; 4 |] in
  let zero = Nx.zeros Nx.float64 [| 32 |] in
  let spectra = Nx.zeros Nx.complex128 [| 7; 5 |] in
  let refused name f = (name, fun () -> ignore (f ())) in
  group "errors"
    [
      cases "every transform refuses an axis out of range" ~name:fst
        [
          refused "fft" (fun () -> Nx.fft ~axis:1 v);
          refused "ifft" (fun () -> Nx.ifft ~axis:(-2) v);
          refused "fftn" (fun () -> Nx.fftn ~axes:[ 2 ] m);
          refused "ifft2" (fun () -> Nx.ifft2 ~axes:[ 0; 2 ] m);
          refused "rfft" (fun () -> Nx.rfft Nx.complex128 ~axis:1 r);
          refused "irfft" (fun () -> Nx.irfft Nx.float64 ~axis:1 v);
          refused "rfftn" (fun () -> Nx.rfftn Nx.complex128 ~axes:[ 2 ] rm);
          refused "irfftn" (fun () -> Nx.irfftn Nx.float64 ~axes:[ -3 ] m);
          refused "hfft" (fun () -> Nx.hfft Nx.float64 ~axis:1 v);
          refused "ihfft" (fun () -> Nx.ihfft Nx.complex128 ~axis:1 r);
          refused "dct" (fun () -> Nx.dct ~axis:1 r);
          refused "dstn" (fun () -> Nx.dstn ~axes:[ 2 ] rm);
          refused "fftshift" (fun () -> Nx.fftshift ~axes:[ 1 ] r);
          refused "ifftshift" (fun () -> Nx.ifftshift ~axes:[ -2 ] r);
        ]
        (fun (_, f) -> raises_invalid_arg f);
      cases "every multi-axis transform refuses s of another length than axes"
        ~name:fst
        [
          refused "fftn" (fun () -> Nx.fftn ~axes:[ 0; 1 ] ~s:[ 2 ] m);
          refused "ifftn" (fun () -> Nx.ifftn ~axes:[ 1 ] ~s:[ 2; 4 ] m);
          refused "fft2" (fun () -> Nx.fft2 ~s:[ 2 ] m);
          refused "rfftn short" (fun () ->
              Nx.rfftn Nx.complex128 ~axes:[ 0; 1 ] ~s:[ 2 ] rm);
          refused "rfftn long" (fun () ->
              Nx.rfftn Nx.complex128 ~axes:[ 1 ] ~s:[ 2; 6 ] rm);
          refused "irfftn" (fun () ->
              Nx.irfftn Nx.float64 ~axes:[ 1 ] ~s:[ 2; 6 ] m);
          refused "irfft2" (fun () -> Nx.irfft2 Nx.float64 ~s:[ 2; 6; 1 ] m);
        ]
        (fun (_, f) -> raises_invalid_arg f);
      cases
        "the 2-D transforms refuse fewer than two dimensions and other than \
         two axes"
        ~name:fst
        [
          refused "fft2 of a vector" (fun () -> Nx.fft2 v);
          refused "ifft2 of a vector" (fun () -> Nx.ifft2 v);
          refused "rfft2 of a vector" (fun () -> Nx.rfft2 Nx.complex128 r);
          refused "irfft2 of a vector" (fun () -> Nx.irfft2 Nx.float64 v);
          refused "fft2 over three axes" (fun () ->
              Nx.fft2 ~axes:[ 0; 1; 2 ] (Nx.ones Nx.complex128 [| 2; 2; 2 |]));
          refused "rfft2 over one axis" (fun () ->
              Nx.rfft2 Nx.complex128 ~axes:[ 0 ] rm);
        ]
        (fun (_, f) -> raises_invalid_arg f);
      cases
        "dct and dst refuse a type outside 1 to 4, a dtype other than float32 \
         and float64, and repeated axes"
        ~name:fst
        [
          refused "dct type 0" (fun () -> Nx.dct ~type_:0 r);
          refused "idst type 5" (fun () -> Nx.idst ~type_:5 r);
          refused "dct of float16" (fun () ->
              Nx.dct (Nx.ones Nx.float16 [| 4 |]));
          refused "idct of bfloat16" (fun () ->
              Nx.idct (Nx.ones Nx.bfloat16 [| 4 |]));
          refused "dctn over 0 and -2" (fun () -> Nx.dctn ~axes:[ 0; -2 ] rm);
          refused "idstn over 1 and 1" (fun () -> Nx.idstn ~axes:[ 1; 1 ] rm);
        ]
        (fun (_, f) -> raises_invalid_arg f);
      cases "hann refuses a length below one" ~name:string_of_int [ 0; -1 ]
        (fun n -> raises_invalid_arg (fun () -> Nx.hann Nx.float64 n));
      cases
        "stft refuses a window below one or past the last axis, a step below \
         one, a 0-d signal and a taper of another shape"
        ~name:fst
        [
          refused "window 0" (fun () -> Nx.stft Nx.complex128 ~window:0 zero);
          refused "window 33" (fun () -> Nx.stft Nx.complex128 ~window:33 zero);
          refused "step 0" (fun () ->
              Nx.stft Nx.complex128 ~window:8 ~step:0 zero);
          refused "0-d" (fun () ->
              Nx.stft Nx.complex128 ~window:1 (Nx.scalar Nx.float64 1.));
          refused "taper of 4 for 8" (fun () ->
              Nx.stft Nx.complex128 ~window:8 ~win:(Nx.hann Nx.float64 4) zero);
          refused "taper of 2 by 4" (fun () ->
              Nx.stft Nx.complex128 ~window:8
                ~win:(Nx.ones Nx.float64 [| 2; 4 |])
                zero);
        ]
        (fun (_, f) -> raises_invalid_arg f);
      cases
        "istft refuses a step outside [1, window], fewer than two dimensions, \
         other than window / 2 + 1 bins, a taper of another shape and a length \
         below one"
        ~name:fst
        [
          refused "step 9 for window 8" (fun () ->
              Nx.istft Nx.float64 ~window:8 ~step:9 spectra);
          refused "step 0" (fun () ->
              Nx.istft Nx.float64 ~window:8 ~step:0 spectra);
          refused "a vector" (fun () ->
              Nx.istft Nx.float64 ~window:8 ~step:4
                (Nx.zeros Nx.complex128 [| 5 |]));
          refused "5 bins for window 12" (fun () ->
              Nx.istft Nx.float64 ~window:12 ~step:4 spectra);
          refused "taper of 4 for 8" (fun () ->
              Nx.istft Nx.float64 ~window:8 ~step:4 ~win:(Nx.hann Nx.float64 4)
                spectra);
          refused "length 0" (fun () ->
              Nx.istft Nx.float64 ~window:8 ~step:4 ~length:0 spectra);
        ]
        (fun (_, f) -> raises_invalid_arg f);
    ]

let () =
  exit
    (run "nx fft"
       [
         complex_transforms;
         multi_axis_complex;
         real_transforms;
         multi_axis_real;
         hermitian_transforms;
         trig_transforms;
         frequencies;
         shifts;
         short_time;
         lengths;
         errors;
       ])
