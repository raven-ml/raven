(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reductions, scans and contractions lowered, against nx's eager result on the
   same operands, or a float64 reference: bit for bit where the lowering is
   exact, within the bound Lower_reduce states elsewhere. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim
module K = Nx_kernel

(* Elements

   A case's elements are int64 raws: a float's bits, an integer's value modulo
   its width, a boolean's low bit, and a complex number's two parts as float64
   bits. *)

let numel s = Array.fold_left ( * ) 1 s

(* A float format's exponent and fraction widths. *)
let format (type s) (dt : (float, s) D.t) =
  match dt with
  | D.Float64 -> (11, 52)
  | D.Float32 -> (8, 23)
  | D.Float16 -> (5, 10)
  | D.Bfloat16 -> (8, 7)
  | D.Float8_e4m3fn -> (4, 3)
  | D.Float8_e5m2 -> (5, 2)
  | D.Float4_e2m1fn -> (2, 1)

let width (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Int64 | D.Uint64 -> 64
  | D.Int32 | D.Uint32 -> 32
  | D.Int16 | D.Uint16 -> 16
  | D.Int8 | D.Uint8 -> 8
  | D.Int4 | D.Uint4 -> 4
  | _ -> 1

let low w r = Int64.to_int r land ((1 lsl w) - 1)

let signed w r =
  let v = low w r in
  if v >= 1 lsl (w - 1) then v - (1 lsl w) else v

let complex raws n =
  let part i = Int64.float_of_bits raws.(i) in
  Array.init n (fun i -> { Complex.re = part (2 * i); im = part ((2 * i) + 1) })

let value (type v s) (dt : (v, s) D.t) shape (raws : int64 array) :
    (v, s, Nx.host) Nx.t =
  let ints : type w r. (w, r) D.t -> (int64 -> w) -> (w, r, Nx.host) Nx.t =
   fun d f -> Nx.create d shape (Array.map f raws)
  in
  match dt with
  | D.Float64 -> Nx.bitcast dt (ints D.Int64 Fun.id)
  | D.Float32 -> Nx.bitcast dt (ints D.Int32 Int64.to_int32)
  | D.Float16 | D.Bfloat16 -> Nx.bitcast dt (ints D.Uint16 (low 16))
  | D.Float8_e4m3fn | D.Float8_e5m2 -> Nx.bitcast dt (ints D.Uint8 (low 8))
  | D.Float4_e2m1fn -> Nx.bitcast dt (ints D.Uint4 (low 4))
  | D.Int64 -> ints dt Fun.id
  | D.Uint64 -> ints dt Fun.id
  | D.Int32 -> ints dt Int64.to_int32
  | D.Uint32 -> ints dt Int64.to_int32
  | D.Int16 -> ints dt (signed 16)
  | D.Uint16 -> ints dt (low 16)
  | D.Int8 -> ints dt (signed 8)
  | D.Uint8 -> ints dt (low 8)
  | D.Int4 -> ints dt (signed 4)
  | D.Uint4 -> ints dt (low 4)
  | D.Bool -> ints dt (fun r -> Int64.logand r 1L = 1L)
  | D.Bit -> ints dt (fun r -> Int64.logand r 1L = 1L)
  | D.Complex128 -> Nx.create dt shape (complex raws (numel shape))
  | D.Complex64 -> Nx.create dt shape (complex raws (numel shape))

(* The number of raws of [n] elements. *)
let raws_of (D.Any dt) n = if D.is D.Complex dt then 2 * n else n

(* Float classes *)

type cls = Number | Zero | Inf | Nan | Subnormal

let float_class (type s) (dt : (float, s) D.t) r =
  let e, f = format dt in
  let ex = low e (Int64.shift_right_logical r f) in
  let fr = Int64.logand r (Int64.pred (Int64.shift_left 1L f)) in
  let top = (1 lsl e) - 1 in
  let zero () = if fr = 0L then Zero else Subnormal in
  match dt with
  | D.Float4_e2m1fn -> if ex = 0 then zero () else Number
  | D.Float8_e4m3fn ->
      if ex = top && fr = 7L then Nan else if ex = 0 then zero () else Number
  | _ ->
      if ex = top then if fr = 0L then Inf else Nan
      else if ex = 0 then zero ()
      else Number

(* A NaN whose quiet bit, its fraction's top bit, is clear. *)
let signalling (type s) (dt : (float, s) D.t) r =
  let _, f = format dt in
  float_class dt r = Nan
  && (match dt with D.Float8_e4m3fn -> false | _ -> true)
  && Int64.logand r (Int64.shift_left 1L (f - 1)) = 0L

let negative (type s) (dt : (float, s) D.t) r =
  let e, f = format dt in
  Int64.logand r (Int64.shift_left 1L (e + f)) <> 0L

(* Generators *)

let among xs = Gen.one_of (List.map Gen.constant xs)

(* [float_raw ~special ~spread ~subnormal dt] draws a float of [dt] whose
   finite numbers lie within [2^±spread], with zeros, infinities and NaNs of
   any sign and payload, of weights [2 special], [special] and [2 special], and
   subnormals of weight [subnormal] against the numbers' 10. *)
let float_raw (type s) ?(special = 1) ~spread ~subnormal (dt : (float, s) D.t)
    =
  let open Gen in
  let e, f = format dt in
  let bias = (1 lsl (e - 1)) - 1 and top = (1 lsl e) - 1 in
  let mask = Int64.pred (Int64.shift_left 1L f) in
  let field sign ex fr =
    let s = if sign then Int64.shift_left 1L (e + f) else 0L in
    Int64.(logor s (logor (shift_left (of_int ex) f) fr))
  in
  let* cls =
    frequency
      [
        (10, constant Number);
        (2 * special, constant Zero);
        (special, constant Inf);
        (2 * special, constant Nan);
        (subnormal, constant Subnormal);
      ]
  in
  let* sign = bool in
  let* ex = int_range (max 1 (bias - spread)) (min (top - 1) (bias + spread)) in
  let+ fr = map (Int64.logand mask) int64 in
  let some fr = if fr = 0L then 1L else fr in
  match (cls, dt) with
  | Number, _ | (Inf | Nan), D.Float4_e2m1fn | Inf, D.Float8_e4m3fn ->
      field sign ex fr
  | Zero, _ -> field sign 0 0L
  | Subnormal, _ -> field sign 0 (some fr)
  | Inf, _ -> field sign top 0L
  | Nan, D.Float8_e4m3fn -> field sign top mask
  | Nan, _ -> field sign top (some fr)

(* [int_raw dt] draws an integer of [dt]: its extremes, small values, or any
   bits. *)
let int_raw (type v s) (dt : (v, s) D.t) =
  let open Gen in
  let w = width dt in
  let lo, hi =
    match (w, D.is D.Signed dt) with
    | 64, true -> (Int64.min_int, Int64.max_int)
    | 64, false -> (0L, -1L)
    | w, true ->
        (Int64.of_int (-(1 lsl (w - 1))), Int64.of_int ((1 lsl (w - 1)) - 1))
    | w, false -> (0L, Int64.of_int ((1 lsl w) - 1))
  in
  if w = 1 then map Int64.of_int (int_range 0 1)
  else
    frequency
      [
        (1, among [ lo; hi ]);
        (3, map Int64.of_int (int_range (-3) 3));
        (1, int64);
      ]

let raw ?special ~spread ~subnormal (D.Any dt) =
  match D.kind dt with
  | D.Float -> float_raw ?special ~spread ~subnormal dt
  | D.Complex -> float_raw ~spread ~subnormal D.Float64
  | _ -> int_raw dt

let extent =
  Gen.frequency
    [ (1, Gen.constant 0); (2, Gen.constant 1); (6, Gen.int_range 2 4) ]

let shape =
  let open Gen in
  let* r = int_range 1 3 in
  map Array.of_list (list ~size:(constant r) extent)

(* Cases *)

type view = Plain | Transposed | Flipped | Broadcast

type case = {
  dt : D.any;
  m : K.monoid;
  shape : int array;
  view : view;
  axes : int list;
  raws : int64 array;
  carry : int64 array option;
}

(* The shape of the array a view of shape [s] reads. *)
let base view s =
  match view with
  | Plain | Flipped -> s
  | Transposed -> Array.of_list (List.rev (Array.to_list s))
  | Broadcast -> Array.mapi (fun i n -> if i = 0 then 1 else n) s

let viewed view s x =
  match view with
  | Plain -> x
  | Transposed -> Nx.transpose x
  | Flipped -> Nx.flip x
  | Broadcast -> Nx.broadcast_to s x

(* The shape of a scan's carry along [axis]. *)
let carried s axis = Array.mapi (fun i n -> if i = axis then 1 else n) s

let monoid_name : K.monoid -> string = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

let pp_case ppf c =
  let (D.Any dt) = c.dt in
  let ints a = String.concat ";" (List.map string_of_int a) in
  let raws a =
    String.concat ";" (Array.to_list (Array.map (Printf.sprintf "%Lx") a))
  in
  Format.fprintf ppf "%s %s [%s] %s axes [%s] raws [%s]%s" (monoid_name c.m)
    (D.name dt)
    (ints (Array.to_list c.shape))
    (match c.view with
    | Plain -> "plain"
    | Transposed -> "transposed"
    | Flipped -> "flipped"
    | Broadcast -> "broadcast")
    (ints c.axes) (raws c.raws)
    (match c.carry with None -> "" | Some a -> " carry [" ^ raws a ^ "]")

(* [case ~spread ~carry dts ms] draws a case of a dtype of the weighted [dts]
   and a monoid of [ms], over one axis where [scan]. *)
let case ?(spread = 4) ?(subnormal = 1) ?(scan = false) dts ms =
  let open Gen in
  let* dt = frequency (List.map (fun (w, d) -> (w, constant d)) dts) in
  let* m = among ms in
  let* s = shape in
  let* view = among [ Plain; Transposed; Flipped; Broadcast ] in
  let r = Array.length s in
  let* keep = list ~size:(constant r) bool in
  let* first = int_range 0 (r - 1) in
  let axes =
    if scan then [ first ]
    else
      List.filter (fun a -> a = first || List.nth keep a) (List.init r Fun.id)
  in
  let* raws =
    array
      ~size:(constant (raws_of dt (numel (base view s))))
      (raw ~spread ~subnormal dt)
  in
  let+ carry =
    if not scan then constant None
    else
      let c = K.carrier dt in
      let n = raws_of c (numel (carried s first)) in
      frequency
        [
          (1, constant None);
          ( 1,
            map Option.some
              (array ~size:(constant n) (raw ~spread ~subnormal c)) );
        ]
  in
  { dt; m; shape = s; view; axes; raws; carry }

let case ?spread ?subnormal ?scan dts ms =
  Gen.with_pp pp_case (case ?spread ?subnormal ?scan dts ms)

let operand (type v s) (dt : (v, s) D.t) c : (v, s, Nx.host) Nx.t =
  viewed c.view c.shape (value dt (base c.view c.shape) c.raws)

let floats =
  D.
    [
      Any Float64;
      Any Float32;
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
    ]

let ints =
  D.
    [
      Any Int64;
      Any Uint64;
      Any Int32;
      Any Uint32;
      Any Int16;
      Any Uint16;
      Any Int8;
      Any Uint8;
      Any Int4;
      Any Uint4;
    ]

let complexes = D.[ Any Complex64; Any Complex128 ]
let weigh w ds = List.map (fun d -> (w, d)) ds

(* Covers *)

let is_float (D.Any dt) = D.is D.Float dt

let float_raws c (f : 's. (float, 's) D.t -> int64 -> bool) =
  let (D.Any dt) = c.dt in
  match D.kind dt with
  | D.Float -> List.filter (f dt) (Array.to_list c.raws)
  | _ -> []

let cover_floats c =
  let has (f : 's. (float, 's) D.t -> int64 -> bool) = float_raws c f <> [] in
  let nans = float_raws c (fun dt r -> float_class dt r = Nan) in
  cover "a NaN term" (nans <> []);
  cover "NaNs of two bit patterns"
    (List.length (List.sort_uniq compare nans) > 1);
  cover "a signalling NaN" (has signalling);
  cover "a negative NaN"
    (has (fun dt r -> float_class dt r = Nan && negative dt r));
  cover "both zeros"
    (has (fun dt r -> float_class dt r = Zero && negative dt r)
    && has (fun dt r -> float_class dt r = Zero && not (negative dt r)));
  cover "an infinity" (has (fun dt r -> float_class dt r = Inf))

let cover_shapes c =
  cover "a strided or broadcast view" (c.view <> Plain);
  cover "a zero-size operand" (numel c.shape = 0);
  cover "a one-element operand" (numel c.shape = 1);
  cover "several axes" (List.length c.axes > 1)

(* Operations *)

let identity dt = K.Prog.v ~ins:[| D.Any dt |] [| In 0 |] ~outs:[| 0 |]

let reduce_op m axes x =
  let dt = Nx.dtype x in
  P.Reduce
    {
      layout = Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x));
      axes = Iarray.of_list axes;
      prog = identity dt;
      reductions = [ Monoid (m, 0, dt) ];
      loads = Iarray.of_list [ P.Plain x ];
    }

let scan_op m axis x carry =
  let dt = Nx.dtype x in
  P.Scan
    {
      layout = Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x));
      axis;
      prog = identity dt;
      reduction = Monoid (m, 0, dt);
      loads = Iarray.of_list [ P.Plain x ];
      carry;
    }

(* [results c] is the eager and the lowered elements of [c]'s reduction or
   scan. *)
let results (type v s) (dt : (v, s) D.t) ~scan c : v array * v array =
  let x = operand dt c in
  if scan then
    let axis = List.hd c.axes in
    let carry =
      Option.map
        (fun raws ->
          let (D.Any cd) = K.carrier (D.Any dt) in
          P.Any (value cd (carried c.shape axis) raws))
        c.carry
    in
    let op () = P.eval ~by:"t" (scan_op c.m axis x carry) in
    (Nx.to_array (op ()), Lowered.run op)
  else
    let op () = fst (P.eval ~by:"t" (reduce_op c.m c.axes x)) in
    (Nx.to_array (op ()), Lowered.run op)

(* Comparisons *)

let hex dt v =
  let b = K.Prog.bits dt v in
  let n = String.length b in
  String.concat ""
    (List.init n (fun i -> Printf.sprintf "%02x" (Char.code b.[n - 1 - i])))

let same_bits dt eager lowered =
  equal (array string) (Array.map (hex dt) eager) (Array.map (hex dt) lowered)

(* Whether [v] is a NaN, or a complex number with a NaN part. *)
let is_nan (type v s) (dt : (v, s) D.t) (v : v) =
  match D.kind dt with
  | D.Float -> Float.is_nan v
  | D.Complex -> Float.is_nan v.Complex.re || Float.is_nan v.im
  | _ -> false

(* Whether [v] is a NaN, or a complex number of two NaN parts. *)
let all_nan (type v s) (dt : (v, s) D.t) (v : v) =
  match D.kind dt with
  | D.Complex -> Float.is_nan v.Complex.re && Float.is_nan v.im
  | _ -> is_nan dt v

(* [same_values dt eager lowered] is [same_bits] but that a NaN result may be
   any NaN, which for a complex extreme has two NaN parts. *)
let same_values dt eager lowered =
  let shown nan v = if nan v then "nan" else hex dt v in
  equal (array string)
    (Array.map (shown (is_nan dt)) eager)
    (Array.map (shown (all_nan dt)) lowered)

let float_kind x =
  if Float.is_nan x then "nan"
  else if x = Float.infinity then "+inf"
  else if x = Float.neg_infinity then "-inf"
  else if x = 0. && Float.sign_bit x then "-0"
  else if x = 0. then "+0"
  else "number"

(* The bits of a float or complex dtype's parts. *)
let part_bits (type v s) (dt : (v, s) D.t) : float -> string =
  match dt with
  | D.Complex64 -> hex D.Float32
  | D.Complex128 -> hex D.Float64
  | _ -> (
      match D.kind dt with
      | D.Float -> hex dt
      | _ -> invalid_arg "part_bits: not a float")

(* [near ~bits ~zeros ~bound e l] checks [l] against [e]: a NaN where [e] is
   one, the same infinity, the same zero where [zeros], and otherwise within
   [bound], inclusive. A zero against a number is within the bound or not, as
   any other difference. *)
let near ?(zeros = true) ~bits ~bound e l =
  let msg = Printf.sprintf "%h (%s) against %h (%s)" e (bits e) l (bits l) in
  let unsigned k = if zeros || (k <> "+0" && k <> "-0") then k else "0" in
  let k = unsigned (float_kind e) and kl = unsigned (float_kind l) in
  let number k = k = "number" || k = "+0" || k = "-0" || k = "0" in
  if not (number k && number kl && (k = "number" || kl = "number")) then
    equal ~msg string k kl
  else less ~msg float_exact ~than:(Float.succ bound) (Float.abs (e -. l))

(* Bounds

   [γ(k) = k u / (1 - k u)]. A result of [dt] narrower than its carrier rounds
   once more on each side: [u_dt] of each and half a subnormal step. *)

let gamma u k =
  let k = Float.of_int (max 0 k) in
  k *. u /. (1. -. (k *. u))

let carrier_roundoff (D.Any dt) =
  match dt with
  | D.Float64 | D.Complex128 -> Float.ldexp 1. (-53)
  | _ -> Float.ldexp 1. (-24)

let rounding (type s) (dt : (float, s) D.t) e l =
  match dt with
  | D.Float64 | D.Float32 -> 0.
  | _ ->
      let ex, f = format dt in
      let tiny = Float.ldexp 1. (2 - (1 lsl (ex - 1)) - f) in
      (Float.ldexp 1. (-(f + 1)) *. (Float.abs e +. Float.abs l)) +. tiny

(* [parts dt xs] is the elements [xs] of the float or complex [dt] as float
   arrays: [[xs]], or their real parts then their imaginary parts. *)
let parts (type v s) (dt : (v, s) D.t) (xs : v array) : float array list =
  match D.kind dt with
  | D.Float -> [ xs ]
  | D.Complex ->
      [
        Array.map (fun z -> z.Complex.re) xs;
        Array.map (fun z -> z.Complex.im) xs;
      ]
  | _ -> invalid_arg "parts: not a float"

(* [parts64 x] is the float or complex [x] as float64 values: [[x]], or its real
   parts, then its imaginary parts. *)
let parts64 (type v s) (x : (v, s, Nx.host) Nx.t) =
  match D.kind (Nx.dtype x) with
  | D.Float -> [ Nx.cast D.Float64 x ]
  | D.Complex -> [ Nx.real D.Float64 x; Nx.imag D.Float64 x ]
  | _ -> invalid_arg "parts64: not a float"

let rounding_of (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Float -> rounding dt | _ -> fun _ _ -> 0.

(* A [Max] or [Min] with a result element and no term is refused. *)
let refused c =
  let kept =
    List.filteri (fun a _ -> not (List.mem a c.axes)) (Array.to_list c.shape)
  in
  (c.m = Max || c.m = Min)
  && List.exists (fun a -> c.shape.(a) = 0) c.axes
  && List.for_all (fun n -> n > 0) kept

let exact ~scan c =
  let (D.Any dt) = c.dt in
  let e, l = results dt ~scan c in
  same_values dt e l

(* Each part of the eager and the lowered results of a float sum or product of
   at most [n] terms, the carry's included, is within [γ(n - 1) Σ|x|] of the
   exact sum, a relative [γ(n - 1)] of the exact product: [Σ|x|] is eager's
   float64 sum of the terms' magnitudes. *)
let bounded ~scan c =
  let (D.Any dt) = c.dt in
  let e, l = results dt ~scan c in
  let axis = List.hd c.axes in
  let xs = parts64 (operand dt c) in
  let carries =
    match c.carry with
    | None -> List.map (fun _ -> None) xs
    | Some raws ->
        let (D.Any cd) = K.carrier c.dt in
        List.map Option.some (parts64 (value cd (carried c.shape axis) raws))
  in
  let magnitudes x carry =
    let x = Nx.abs x in
    let m = if scan then Nx.cumsum ~axis x else Nx.sum ~axes:c.axes x in
    let m = match carry with None -> m | Some k -> Nx.add m (Nx.abs k) in
    Nx.to_array m
  in
  let terms = List.fold_left (fun n a -> n * c.shape.(a)) 1 c.axes in
  let n = if c.carry = None then terms else terms + 1 in
  let g = gamma (carrier_roundoff c.dt) (n - 1) in
  let bits = part_bits dt and round = rounding_of dt in
  let check (e, l) (x, carry) =
    let mags = magnitudes x carry in
    Array.iteri
      (fun o e ->
        let l = l.(o) in
        let bound =
          match c.m with
          | Prod -> 2.01 *. g *. Float.max (Float.abs e) (Float.abs l)
          | _ -> 2. *. g *. mags.(o)
        in
        near ~bits ~bound:(bound +. round e l) e l)
      e
  in
  List.iter2 check
    (List.combine (parts dt e) (parts dt l))
    (List.combine xs carries)

let is_bool (D.Any dt) = D.is D.Boolean dt
let is_complex (D.Any dt) = D.is D.Complex dt
let reduced_empty c = List.exists (fun a -> c.shape.(a) = 0) c.axes
let cover_monoids c ms = List.iter (fun m -> cover (monoid_name m) (c.m = m)) ms

(* Examples every run takes: a one-element and a zero-size operand, and for
   float32 and float16 five rows: NaNs of both signs and kinds; infinities of
   both signs before a signalling NaN; zeros of both signs; all [-0]; and
   infinities of both signs, a NaN made from numbers by a sum or a product. A
   scan's carries are [1], a signalling NaN, [+0], [-0] and [-∞]. *)

(* Complex numbers as raws: each part's float64 bits. *)
let complex_raws zs =
  Array.concat
    (List.map
       (fun (re, im) -> [| Int64.bits_of_float re; Int64.bits_of_float im |])
       zs)

(* For complex64, five rows: equal real parts; a NaN imaginary part beside a
   [+0] real part before a NaN real part; zeros of both signs; [-∞] real parts;
   infinities and a number of two NaN parts. A scan's carries are numbers, a NaN
   real part and infinities. *)
let complex_specials =
  let nan = Int64.float_of_bits 0x7ff4000000000001L in
  let qnan = Int64.float_of_bits 0xfff8000000000003L in
  let inf = Float.infinity in
  ( complex_raws
      [
        (1., 2.);
        (1., 3.);
        (-0., 4.);
        (0., -4.);
        (0., qnan);
        (inf, 1.);
        (nan, 0.);
        (2., 2.);
        (-0., 1.);
        (0., 1.);
        (-0., 2.);
        (0., 0.);
        (-.inf, -0.);
        (-.inf, 0.);
        (-.inf, -0.);
        (-.inf, 0.);
        (1., inf);
        (1., -.inf);
        (nan, qnan);
        (inf, 0.);
      ],
    complex_raws [ (1., 2.); (0., 0.); (nan, 1.); (-0., -0.); (inf, -.inf) ] )

let special_carry = [| 0x3f800000L; 0x7f800009L; 0L; 0x80000000L; 0xff800000L |]

let specials (D.Any d) =
  match d with
  | D.Float32 ->
      Some
        ( [|
            0x7fc00001L;
            0xff800005L;
            0L;
            0x80000000L;
            0x7f800000L;
            0xff800000L;
            0x3f800000L;
            0x7fa00000L;
            0x80000000L;
            0L;
            0x80000000L;
            0xbf800000L;
            0x80000000L;
            0x80000000L;
            0x80000000L;
            0x80000000L;
            0x7f800000L;
            0xff800000L;
            0x3f800000L;
            0L;
          |],
          special_carry )
  | D.Float16 ->
      Some
        ( [|
            0x7e01L;
            0xfc05L;
            0L;
            0x8000L;
            0x7c00L;
            0xfc00L;
            0x3c00L;
            0x7d00L;
            0x8000L;
            0L;
            0x8000L;
            0xbc00L;
            0x8000L;
            0x8000L;
            0x8000L;
            0x8000L;
            0x7c00L;
            0xfc00L;
            0x3c00L;
            0L;
          |],
          special_carry )
  | D.Complex64 -> Some complex_specials
  | _ -> None

let examples ?(scan = false) dts ms =
  let ex dt m shape axes raws carry =
    { dt; m; shape; view = Plain; axes; raws; carry }
  in
  let each dt (m : K.monoid) =
    let zero =
      if (m = Max || m = Min) && not scan then [| 0; 2 |] else [| 2; 0 |]
    in
    let special =
      match specials dt with
      | None -> []
      | Some (raws, carry) ->
          [ ex dt m [| 5; 4 |] [ 1 ] raws (if scan then Some carry else None) ]
    in
    ex dt m [| 1 |] [ 0 ] (Array.make (raws_of dt 1) 1L) None
    :: ex dt m zero [ 1 ] [||] None
    :: special
  in
  List.concat_map (fun dt -> List.concat_map (each dt) ms) dts

(* Reductions *)

let extremes =
  let dts =
    weigh 3 floats @ weigh 2 ints
    @ weigh 6 D.[ Any Bool; Any Bit ]
    @ weigh 4 complexes
  in
  prop ~count:30
    ~examples:
      (examples
         D.[ Any Float32; Any Float16; Any Bool; Any Int8; Any Complex64 ]
         [ Max; Min ])
    "extremes give eager's values"
    (case dts [ Max; Min ])
    (fun c ->
      assume (not (refused c));
      cover_monoids c [ Max; Min ];
      cover_floats c;
      cover_shapes c;
      cover "a boolean" (is_bool c.dt);
      cover "a complex number" (is_complex c.dt);
      cover "an integer" (not (is_float c.dt || is_bool c.dt));
      exact ~scan:false c)

let int_folds =
  prop ~count:15
    ~examples:(examples D.[ Any Int8 ] [ Sum; Prod ])
    "integer sums and products wrap as eager's"
    (case (weigh 1 ints) [ Sum; Prod ])
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "an empty reduced extent" (reduced_empty c);
      cover_shapes c;
      exact ~scan:false c)

let float_sums =
  prop ~count:15
    ~examples:(examples D.[ Any Float32; Any Float16 ] [ Sum ])
    "float sums hold their bound"
    (case (weigh 1 floats) [ Sum ])
    (fun c ->
      cover_floats c;
      cover "an empty reduced extent" (reduced_empty c);
      cover "a narrow float"
        (not (List.mem c.dt D.[ Any Float32; Any Float64 ]));
      bounded ~scan:false c)

(* Products of numbers of one binade: no overflow or underflow. *)
let float_products =
  prop ~count:12
    ~examples:(examples D.[ Any Float32 ] [ Prod ])
    "float products hold their bound"
    (case ~spread:0 ~subnormal:0 (weigh 1 floats) [ Prod ])
    (fun c ->
      cover "a NaN term"
        (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
      cover "an empty reduced extent" (reduced_empty c);
      bounded ~scan:false c)

let complex_sums =
  prop ~count:8
    ~examples:(examples D.[ Any Complex64 ] [ Sum ])
    "complex sums add their parts"
    (case (weigh 1 complexes) [ Sum ])
    (fun c ->
      cover "complex64" (c.dt = D.Any D.Complex64);
      bounded ~scan:false c)

(* Scans *)

let extreme_scans =
  let dts =
    weigh 3 floats @ weigh 2 ints
    @ weigh 6 D.[ Any Bool; Any Bit ]
    @ weigh 4 D.[ Any Complex64 ]
  in
  prop ~count:24
    ~examples:
      (examples ~scan:true
         D.[ Any Float32; Any Float16; Any Bool; Any Complex64 ]
         [ Max; Min ])
    "extreme scans give eager's values"
    (case ~scan:true dts [ Max; Min ])
    (fun c ->
      cover_monoids c [ Max; Min ];
      cover "a carry" (c.carry <> None);
      cover "a NaN term"
        (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
      cover "a boolean" (is_bool c.dt);
      cover "a complex number" (is_complex c.dt);
      exact ~scan:true c)

let int_scans =
  prop ~count:12
    ~examples:(examples ~scan:true D.[ Any Int8 ] [ Sum; Prod ])
    "integer running sums and products wrap as eager's"
    (case ~scan:true (weigh 1 ints) [ Sum; Prod ])
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "a carry" (c.carry <> None);
      exact ~scan:true c)

let float_scans =
  prop ~count:12
    ~examples:(examples ~scan:true D.[ Any Float32; Any Float16 ] [ Sum; Prod ])
    "float running sums and products hold their bound"
    (case ~scan:true ~spread:0 ~subnormal:0 (weigh 1 floats) [ Sum; Prod ])
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "a carry" (c.carry <> None);
      bounded ~scan:true c)

let complex_scans =
  prop ~count:5
    ~examples:(examples ~scan:true D.[ Any Complex64 ] [ Sum ])
    "complex running sums add their parts"
    (case ~scan:true (weigh 1 complexes) [ Sum ])
    (fun c ->
      cover "a carry" (c.carry <> None);
      bounded ~scan:true c)

(* Logsumexp, Moments and Arg

   Arg gives eager's values and positions. Logsumexp and Moments, into the
   operand's dtype, hold the bounds Lower_reduce states, plus the rounding
   into a destination narrower than the carrier, against a float64 reference:
   OCaml's [Float] functions on the terms, the special values from
   {!Nx_kernel.type-reduction}. The reference is in error by the reduction's
   bounds at float64's unit roundoff, with glibc's [exp] and [log] within an
   ulp; for float64 the two errors add. *)

(* [E] and [L], the bounds in ulps Lower_real states for [Exp] and [Log] at
   float32 and float64. *)
let exp_ulps = 9.
let log_ulps = 9.
let u64 = Float.ldexp 1. (-53)

(* The loop of [x]'s identity by [r]: reduced along [c]'s axes, or scanned
   along its first where [scan]. *)
let by (type a) ~scan (r : (Nx.host, a) P.reduction) c x () : a =
  let layout = Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x)) in
  let prog = identity (Nx.dtype x) and loads = Iarray.of_list [ P.Plain x ] in
  if scan then
    P.eval ~by:"t"
      (P.Scan
         {
           layout;
           axis = List.hd c.axes;
           prog;
           reduction = r;
           loads;
           carry = None;
         })
  else
    fst
      (P.eval ~by:"t"
         (P.Reduce
            {
              layout;
              axes = Iarray.of_list c.axes;
              prog;
              reductions = [ r ];
              loads;
            }))

let extreme c : K.extreme = if c.m = Max then Max else Min

let arg_law ~scan c =
  let (D.Any dt) = c.dt in
  let op = by ~scan (Arg (extreme c, 0, dt)) c (operand dt c) in
  same_values dt
    (Nx.to_array (fst (op ())))
    (Lowered.run (fun () -> fst (op ())));
  equal (array int64)
    (Nx.to_array (snd (op ())))
    (Lowered.run (fun () -> snd (op ())))

(* [slices ~scan c xs] is the terms of each result of [c] in C order, of the
   elements [xs] of its operand: a reduction's along its axes, a scan's along
   its axis up to the result's index. *)
let slices ~scan c xs =
  let s = c.shape in
  let r = Array.length s in
  let reduced a = if scan then a = List.hd c.axes else List.mem a c.axes in
  let kept = List.filter (fun a -> not (reduced a)) (List.init r Fun.id) in
  let index i =
    let idx = Array.make r 0 and rest = ref i in
    for a = r - 1 downto 0 do
      idx.(a) <- !rest mod s.(a);
      rest := !rest / s.(a)
    done;
    idx
  in
  let key idx = List.fold_left (fun k a -> (k * s.(a)) + idx.(a)) 0 kept in
  let groups = Array.make (List.fold_left (fun n a -> n * s.(a)) 1 kept) [] in
  Array.iteri
    (fun i x ->
      let k = key (index i) in
      groups.(k) <- x :: groups.(k))
    xs;
  let groups = Array.map List.rev groups in
  if not scan then groups
  else
    Array.init (numel s) (fun i ->
        let idx = index i in
        List.filteri (fun j _ -> j <= idx.(List.hd c.axes)) groups.(key idx))

let lse ts =
  let has p = List.exists p ts in
  if has Float.is_nan then Float.nan
  else if has (Float.equal Float.infinity) then Float.infinity
  else if List.for_all (Float.equal Float.neg_infinity) ts then
    Float.neg_infinity
  else
    let m = List.fold_left Float.max Float.neg_infinity ts in
    m +. Float.log (List.fold_left (fun s t -> s +. Float.exp (t -. m)) 0. ts)

let moments ts =
  let n = Float.of_int (List.length ts) in
  let sum f = List.fold_left (fun s t -> s +. f t) 0. ts in
  let mean = sum Fun.id /. n in
  if not (Float.is_finite mean) then (mean, Float.nan)
  else
    let d t = t -. mean in
    let s1 = sum d and s2 = sum (fun t -> d t *. d t) in
    (mean, Float.max 0. ((s2 -. (s1 *. (s1 /. n))) /. n))

let lse_bound u ~e ~l n r =
  let n = Float.of_int (max 1 n) in
  ((((1. +. Float.exp (-1.)) *. (n -. 1.)) +. (2. *. e) +. 1.
   +. (((2. *. l) +. 1.) *. Float.log n))
  *. u)
  +. (u *. Float.abs r)

let mean_bound u n mag = Float.of_int (n + 1) *. u *. mag

(* [depth n] is the network's [D] at a result of [n] terms:
   [max 0 (2⌈log₂ n⌉ - 1)]. *)
let depth n =
  let rec log k p = if p >= n then k else log (k + 1) (2 * p) in
  Float.of_int (max 0 ((2 * log 0 1) - 1))

let lse_scan_bound u ~e ~l n r =
  let d = depth n and n = Float.of_int (max 1 n) in
  ((((2. *. e) +. 3.) *. d) +. (((2. *. l) +. 2.) *. Float.log n)) *. u
  +. (u *. Float.abs r)

let mean_scan_bound u n range mean =
  (((4. *. depth n) +. 1.) *. u *. range) +. (u *. Float.abs mean)

let var_scan_bound u n range v =
  let d = depth n in
  (((7. *. d) +. 1.) *. u *. v)
  +. (((4. *. d *. d) -. (2. *. d)) *. u *. range *. Float.sqrt v)

let var_bound u n mag v =
  let b = mean_bound u n mag and n = Float.of_int n in
  ((n +. 4.) *. u *. v)
  +. (2. *. n *. u *. b *. Float.sqrt v)
  +. (3. *. (n +. 2.) *. u *. b *. b)

(* Each part's error is first order: [1.01] holds the rest. A rounding below the
   least normal errs by up to half the least subnormal [eta]. *)
let close (type s) (dt : (float, s) D.t) ~eta n ~ours ~reference e l =
  let tiny = Float.of_int (2 * (n + 4)) *. eta in
  let bound = (1.01 *. (ours +. reference)) +. tiny in
  near ~zeros:false ~bits:(hex dt) ~bound e l

let cover_slices tss =
  let any p = Array.exists p tss in
  let finite ts = List.filter (fun t -> Float.is_finite t && t <> 0.) ts in
  let spread ts =
    match List.map Float.abs (finite ts) with
    | [] -> 1.
    | a :: r ->
        List.fold_left Float.max a r /. List.fold_left Float.min a r
  in
  cover "a slice of -∞ alone"
    (any (fun ts ->
         ts <> [] && List.for_all (Float.equal Float.neg_infinity) ts));
  cover "a slice of one term" (any (fun ts -> List.length ts = 1));
  cover "a slice of over 1000 finite terms"
    (any (fun ts -> List.length ts > 1000 && List.for_all Float.is_finite ts));
  cover "a slice spread past 2^20" (any (fun ts -> spread ts > 0x1p20));
  cover "a NaN term" (any (List.exists Float.is_nan));
  cover "an infinity"
    (any (List.exists (fun t -> Float.abs t = Float.infinity)));
  cover "a zero" (any (List.exists (Float.equal 0.)))

let float_law (type s) ~scan which c (dt : (float, s) D.t) =
  let (K.Any_float cd) = K.float_carrier dt in
  let x = operand dt c in
  let tss = slices ~scan c (Nx.to_array (Nx.cast D.Float64 x)) in
  cover_slices tss;
  cover "a narrow destination"
    (not (List.mem c.dt D.[ Any Float32; Any Float64 ]));
  let u = carrier_roundoff c.dt in
  let eta = match cd with D.Float64 -> 0x1p-1074 | _ -> 0x1p-149 in
  (* The references rounded into [dt], as the results are. *)
  let rounded es =
    let es = Nx.create D.Float64 [| Array.length es |] es in
    Nx.to_array (Nx.cast D.Float64 (Nx.cast dt es))
  in
  let close n e l ~ours ~reference =
    close dt ~eta n e l ~ours:(ours +. rounding dt e l) ~reference
  in
  match which with
  | `Logsumexp ->
      let l = Lowered.run (by ~scan (Logsumexp (0, dt)) c x) in
      let es = rounded (Array.map lse tss) in
      Array.iteri
        (fun i ts ->
          let n = List.length ts and e = es.(i) in
          let ours =
            if scan then lse_scan_bound u ~e:exp_ulps ~l:log_ulps n e
            else lse_bound u ~e:exp_ulps ~l:log_ulps n e
          in
          close n e l.(i) ~ours ~reference:(lse_bound u64 ~e:1. ~l:1. n e))
        tss
  | `Moments ->
      let op = by ~scan (Moments (0, dt)) c x in
      let mean = Lowered.run (fun () -> fst (op ())) in
      let var = Lowered.run (fun () -> snd (op ())) in
      let ems = rounded (Array.map (fun ts -> fst (moments ts)) tss) in
      let evs = rounded (Array.map (fun ts -> snd (moments ts)) tss) in
      Array.iteri
        (fun i ts ->
          let n = List.length ts and em = ems.(i) and ev = evs.(i) in
          let mag =
            List.fold_left (fun s t -> s +. Float.abs t) 0. ts /. Float.of_int n
          in
          let range =
            List.fold_left Float.max Float.neg_infinity ts
            -. List.fold_left Float.min Float.infinity ts
          in
          let ours_mean, ours_var =
            if scan then
              (mean_scan_bound u n range em, var_scan_bound u n range ev)
            else (mean_bound u n mag, var_bound u n mag ev)
          in
          close n em mean.(i) ~ours:ours_mean
            ~reference:(mean_bound u64 n mag);
          close n ev var.(i) ~ours:ours_var
            ~reference:(var_bound u64 n mag ev);
          equal ~msg:(Printf.sprintf "variance %h" var.(i)) bool false
            (var.(i) < 0.))
        tss

let floating ~scan which c =
  let (D.Any dt) = c.dt in
  match D.kind dt with
  | D.Float -> float_law ~scan which c dt
  | _ -> invalid_arg "floating: not a float"

(* Rows of -∞ alone; of numbers [2^40] apart; and of a mean far from its
   deviations, [4096.5] and [4097]. *)
let spread_example =
  {
    dt = D.Any D.Float32;
    m = Sum;
    shape = [| 3; 4 |];
    view = Plain;
    axes = [ 1 ];
    raws =
      Array.concat
        [
          Array.make 4 0xff800000L;
          [| 0x3f800000L; 0x21800000L; 0x53800000L; 0xd3800000L |];
          [| 0x45800400L; 0x45800800L; 0x45800400L; 0x45800800L |];
        ];
    carry = None;
  }

(* A row of the [n] terms [sin i] of [dt], [float32] or [float16]. *)
let sine_row dt n =
  let ts = Array.init n (fun i -> Float.sin (Float.of_int i)) in
  let raws =
    match dt with
    | D.Any D.Float32 ->
        Array.map Int64.of_int32
          (Nx.to_array (Nx.bitcast D.Int32 (Nx.create D.Float32 [| n |] ts)))
    | _ ->
        Array.map Int64.of_int
          (Nx.to_array (Nx.bitcast D.Uint16 (Nx.create D.Float16 [| n |] ts)))
  in
  { spread_example with dt; shape = [| 1; n |]; raws }

(* A row of hundreds to thousands of finite terms within [2^±spread], which
   reaches the bounds' growth with the number of terms. *)
let long_row =
  let open Gen in
  let* dt = among floats in
  let* n = int_range 100 2000 in
  let* spread = among [ 0; 2; 30 ] in
  let+ raws =
    array ~size:(constant n) (raw ~special:0 ~spread ~subnormal:0 dt)
  in
  { spread_example with dt; shape = [| 1; n |]; raws }

let float_examples ~scan =
  spread_example
  :: sine_row (D.Any D.Float32) 1500
  :: sine_row (D.Any D.Float16) 300
  :: List.map
       (fun c -> { c with carry = None })
       (examples ~scan D.[ Any Float32; Any Float16; Any Float64 ] [ Sum ])

let float_prop ~scan which name =
  let short = case ~spread:30 ~scan (weigh 1 floats) [ Sum ] in
  prop ~count:10 ~examples:(float_examples ~scan) name
    (Gen.frequency [ (3, short); (1, Gen.with_pp pp_case long_row) ])
    (fun c ->
      let c = { c with carry = None } in
      cover "an empty axis" (List.exists (fun a -> c.shape.(a) = 0) c.axes);
      floating ~scan which c)

let logsumexps =
  float_prop ~scan:false `Logsumexp "a logsumexp holds its bound"

let logsumexp_scans =
  float_prop ~scan:true `Logsumexp "a running logsumexp holds its bound"

let moment_reductions =
  float_prop ~scan:false `Moments "moments hold their bounds"

let moment_scans =
  float_prop ~scan:true `Moments "running moments hold their bounds"

let arg_prop ~scan dts name =
  prop ~count:20
    ~examples:
      (List.map
         (fun c -> { c with carry = None })
         (examples ~scan
            D.[ Any Float32; Any Float16; Any Bool; Any Int8; Any Complex64 ]
            [ Max; Min ]))
    name (case ~scan dts [ Max; Min ])
    (fun c ->
      let c = { c with carry = None } in
      assume (scan || not (refused c));
      cover_monoids c [ Max; Min ];
      cover_floats c;
      if not scan then cover_shapes c;
      cover "a boolean" (is_bool c.dt);
      cover "a complex number" (is_complex c.dt);
      cover "an integer"
        (not (is_float c.dt || is_bool c.dt || is_complex c.dt));
      arg_law ~scan c)

let arg_dts =
  weigh 3 floats @ weigh 2 ints
  @ weigh 4 D.[ Any Bool; Any Bit ]
  @ weigh 3 complexes

let args =
  arg_prop ~scan:false arg_dts "an arg gives eager's values and positions"

let arg_scans =
  arg_prop ~scan:true
    (List.filter (fun (_, d) -> d <> D.Any D.Complex128) arg_dts)
    "a running arg gives eager's values and positions"

(* Contractions

   Each operand's axes are batch, free or contracted, in any order, read through
   a transposed view or not. An operand has at most one free axis, three for
   [Rows]; the result's free axes are [a]'s, then [b]'s, in axis order. A
   scale's blocks divide each of its operand's extents; codes stay below
   [151] or are [255], so that no partial sum overflows. Offsets are drawn
   around the segmented axis, int64's extremes included. *)

type role = B of int | F of int | C of int

type scale =
  | Codes of int array * int64 array
  | Values of D.any * int array * int64 array

(* [Rows (i, j, o)] segments [a]'s free axis [i] by [b]'s free axis [j], of
   extent the groups'. *)
type segments =
  | Rows of int * int * int64 array
  | Contracting of int * int * int64 array

type contraction = {
  cdt : D.any;
  acc : D.any;
  eb : int array;
  ek : int array;
  fa : int array;
  fb : int array;
  ra : role array;
  rb : role array;
  ta : bool;
  tb : bool;
  araws : int64 array;
  braws : int64 array;
  init : int64 array option;
  sa : scale option;
  sb : scale option;
  seg : segments option;
}

let ashape c f r =
  Array.map (function B i -> c.eb.(i) | C i -> c.ek.(i) | F i -> f.(i)) r

let out_shape c =
  match c.seg with
  | None -> Array.concat [ c.eb; c.fa; c.fb ]
  | Some (Rows (_, j, _)) ->
      let fb = List.filteri (fun k _ -> k <> j) (Array.to_list c.fb) in
      Array.concat [ c.eb; c.fa; Array.of_list fb ]
  | Some (Contracting (_, g, _)) -> Array.concat [ [| g |]; c.eb; c.fa; c.fb ]

(* The groups of [Contracting]'s pair or [Rows]' [a] axis, and its extent. *)
let groups c =
  match c.seg with
  | None -> (0, 0)
  | Some (Rows (i, j, _)) -> (c.fb.(j), c.fa.(i))
  | Some (Contracting (p, g, _)) -> (g, c.ek.(p))

let offsets_of = function Rows (_, _, o) | Contracting (_, _, o) -> o

(* [member o ~g r] is the group of index [r] by the offsets [o.(0)] to
   [o.(g)], as {!Nx_kernel.type-segments} states. *)
let member o ~g r =
  let r = Int64.of_int r in
  let rec find e =
    if e = g then None
    else
      let rec below j = j > g || (r < o.(j) && below (j + 1)) in
      if Int64.max o.(0) o.(e) <= r && below (e + 1) then Some e
      else find (e + 1)
  in
  find 0

(* The offsets of each batch index. *)
let batches c =
  let g, _ = groups c in
  match c.seg with
  | None -> []
  | Some s ->
      let o = offsets_of s in
      List.init (numel c.eb) (fun i -> Array.sub o (i * (g + 1)) (g + 1))

let pairs c pick n =
  let pos r x = Option.get (Array.find_index (( = ) x) r) in
  Iarray.of_list (List.init n (fun i -> (pos c.ra (pick i), pos c.rb (pick i))))

let pp_contraction ppf c =
  let name (D.Any d) = D.name d in
  let ints a = String.concat ";" (Array.to_list (Array.map string_of_int a)) in
  let role = function
    | B i -> "b" ^ string_of_int i
    | C i -> "k" ^ string_of_int i
    | F i -> "f" ^ string_of_int i
  in
  let roles r = String.concat ";" (Array.to_list (Array.map role r)) in
  let raws a =
    String.concat ";" (Array.to_list (Array.map (Printf.sprintf "%Lx") a))
  in
  Format.fprintf ppf
    "%s acc %s a [%s]%s [%s] b [%s]%s [%s] raws a [%s] b [%s]%s" (name c.cdt)
    (name c.acc) (roles c.ra)
    (if c.ta then " transposed" else "")
    (ints (ashape c c.fa c.ra))
    (roles c.rb)
    (if c.tb then " transposed" else "")
    (ints (ashape c c.fb c.rb))
    (raws c.araws) (raws c.braws)
    (match c.init with None -> "" | Some r -> " init [" ^ raws r ^ "]");
  let scale side = function
    | None -> ()
    | Some (Codes (s, r)) ->
        Format.fprintf ppf " %s codes [%s] [%s]" side (ints s) (raws r)
    | Some (Values (d, s, r)) ->
        Format.fprintf ppf " %s %s [%s] [%s]" side (name d) (ints s) (raws r)
  in
  scale "sa" c.sa;
  scale "sb" c.sb;
  match c.seg with
  | None -> ()
  | Some (Rows (i, j, o)) ->
      Format.fprintf ppf " rows f%d by f%d [%s]" i j (raws o)
  | Some (Contracting (p, g, o)) ->
      Format.fprintf ppf " pair %d groups %d [%s]" p g (raws o)

type extras = Plain_contraction | Scaled | Segmented

let contraction ?(extras = Plain_contraction) dts accs =
  let open Gen in
  let small =
    frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 3) ]
  in
  let extents n = map Array.of_list (list ~size:(constant n) small) in
  let roles nf eb ek =
    List.init (Array.length eb) (fun i -> B i)
    @ List.init nf (fun i -> F i)
    @ List.init (Array.length ek) (fun i -> C i)
  in
  (* Free axes numbered in axis order, the result's. *)
  let ordered roles =
    let number k = function F _ -> (k + 1, F k) | r -> (k, r) in
    Array.of_list (snd (List.fold_left_map number 0 roles))
  in
  let* cdt = frequency (List.map (fun (w, d) -> (w, constant d)) dts) in
  let* acc = among (accs cdt) in
  let* rows = if extras = Segmented then bool else constant false in
  let segmented = extras = Segmented in
  let count lo hi =
    let* n = int_range lo hi in
    extents n
  in
  let* eb = count 0 1 in
  let* ek = count (if segmented && not rows then 1 else 0) 2 in
  (* [Rows] draws up to two free axes on each side beyond the segmented one. *)
  let lo, hi = if rows then (1, 3) else (0, 1) in
  let* fa = count lo hi in
  let* fb = count lo hi in
  let* ra = map ordered (permutation (roles (Array.length fa) eb ek)) in
  let* rb = map ordered (permutation (roles (Array.length fb) eb ek)) in
  let* ta = bool in
  let* tb = bool in
  let c =
    {
      cdt;
      acc;
      eb;
      ek;
      fa;
      fb;
      ra;
      rb;
      ta;
      tb;
      araws = [||];
      braws = [||];
      init = None;
      sa = None;
      sb = None;
      seg = None;
    }
  in
  let raws s =
    array
      ~size:(constant (raws_of cdt (numel s)))
      (raw ~spread:2 ~subnormal:1 cdt)
  in
  let* araws = raws (ashape c fa ra) in
  let* braws = raws (ashape c fb rb) in
  let offsets g n =
    let entry =
      frequency
        [
          (8, map Int64.of_int (int_range (-1) (n + 1)));
          (1, among [ Int64.min_int; Int64.max_int ]);
        ]
    in
    array ~size:(constant (numel eb * (g + 1))) entry
  in
  let* seg =
    if not segmented then constant None
    else if rows then
      let* i = int_range 0 (Array.length fa - 1) in
      let* j = int_range 0 (Array.length fb - 1) in
      map (fun o -> Some (Rows (i, j, o))) (offsets fb.(j) fa.(i))
    else
      let* p = int_range 0 (Array.length ek - 1) in
      let* g = int_range 0 3 in
      map (fun o -> Some (Contracting (p, g, o))) (offsets g ek.(p))
  in
  let scale s =
    let divisors n =
      if n = 0 then [ 0 ] else List.filter (fun k -> n mod k = 0) [ 1; 2; 3 ]
    in
    let rec each = function
      | [] -> constant []
      | n :: ns ->
          let* k = among (divisors n) in
          let+ ks = each ns in
          k :: ks
    in
    let* blocks = map Array.of_list (each (Array.to_list s)) in
    let k = numel blocks in
    let code =
      frequency
        [
          (1, constant 0L);
          (1, constant 255L);
          (6, map Int64.of_int (int_range 100 150));
        ]
    in
    let codes =
      map (fun r -> Codes (blocks, r)) (array ~size:(constant k) code)
    in
    let values =
      let* d = among D.[ Any Float32; Any Float64; Any Float16 ] in
      map
        (fun r -> Values (d, blocks, r))
        (array ~size:(constant k) (raw ~spread:2 ~subnormal:1 d))
    in
    frequency
      [
        (1, constant None);
        (1, map Option.some codes);
        (1, map Option.some values);
      ]
  in
  let scaled s = if extras = Scaled then scale s else constant None in
  let* sa = scaled (ashape c fa ra) in
  let* sb = scaled (ashape c fb rb) in
  let c = { c with araws; braws; sa; sb; seg } in
  let+ init =
    frequency [ (1, constant None); (1, map Option.some (raws (out_shape c))) ]
  in
  { c with init }

let contraction ?extras dts accs =
  Gen.with_pp pp_contraction (contraction ?extras dts accs)

(* An operand of shape [s], read through a transposed view where [t]. *)
let contracted (type v s) (dt : (v, s) D.t) s t raws : (v, s, Nx.host) Nx.t =
  if not t then value dt s raws
  else Nx.transpose (value dt (Array.of_list (List.rev (Array.to_list s))) raws)

(* A scale's array, its values' magnitudes where [abs]. *)
let scale_of ~abs : scale -> Nx.host P.scale = function
  | Codes (s, r) -> Exponent (Nx.create D.Uint8 s (Array.map Int64.to_int r))
  | Values (D.Any d, s, r) ->
      let v = value d s r in
      Value (if abs then Nx.abs v else v)

let segments_of c : _ K.segments =
  let g, _ = groups c in
  let pos r x = Option.get (Array.find_index (( = ) x) r) in
  match c.seg with
  | Some (Rows (i, j, o)) ->
      let offsets = Nx.create D.Int64 (Array.append c.eb [| g + 1 |]) o in
      Rows { offsets; a_axis = pos c.ra (F i); b_axis = pos c.rb (F j) }
  | Some (Contracting (p, _, o)) ->
      let offsets = Nx.create D.Int64 (Array.append c.eb [| g + 1 |]) o in
      Contracting { offsets; pair = p }
  | None -> invalid_arg "segments_of: no segments"

(* The contraction of [a] and [b] from [init] into [dt], as [c] pairs their
   axes, summed in [acc], scaled and segmented as [c] says, its scales'
   magnitudes where [abs]. *)
let contraction_op ?(abs = false) c ~acc dt a b init =
  let batch = pairs c (fun i -> B i) (Array.length c.eb) in
  let contracting = pairs c (fun i -> C i) (Array.length c.ek) in
  P.Contract
    {
      batch;
      contracting;
      acc;
      out = dt;
      a;
      b;
      init;
      a_scale = Option.map (scale_of ~abs) c.sa;
      b_scale = Option.map (scale_of ~abs) c.sb;
      segments = Option.map (fun _ -> segments_of c) c.seg;
    }

let contract_results (type v s) (dt : (v, s) D.t) c =
  let a = contracted dt (ashape c c.fa c.ra) c.ta c.araws in
  let b = contracted dt (ashape c c.fb c.rb) c.tb c.braws in
  let init = Option.map (value dt (out_shape c)) c.init in
  let op () = P.eval ~by:"t" (contraction_op c ~acc:c.acc dt a b init) in
  (a, b, init, Nx.to_array (op ()), Lowered.run op)

(* Each part's [|init| + Σ|x||y|] over its products of parts, by eager float64
   contractions: [ar·br] and [ai·bi] in the real part, [ar·bi] and [ai·br] in
   the imaginary part. *)
let magnitudes c a b init =
  let abs ps = List.map Nx.abs ps in
  let a = abs (parts64 a) and b = abs (parts64 b) in
  let start p = Option.map (fun i -> List.nth (abs (parts64 i)) p) init in
  let sum x y init =
    P.eval ~by:"t"
      (contraction_op ~abs:true c ~acc:(D.Any D.Float64) D.Float64 x y init)
  in
  match (a, b) with
  | [ a ], [ b ] -> [ sum a b (start 0) ]
  | [ ar; ai ], [ br; bi ] ->
      [
        Nx.add (sum ar br (start 0)) (sum ai bi None);
        Nx.add (sum ar bi (start 1)) (sum ai br None);
      ]
  | _ -> invalid_arg "magnitudes: not a float"

(* Each output's count of products whose operands are both nonzero. Where it has
   none, its sum is of exact zeros, [+0] from a start of [+0]. Elsewhere a zero
   result's sign is open: a product that underflows keeps its sign fused into
   the sum and is [+0] once rounded and added to [+0], and a narrow output
   rounds a tiny sum of either sign to zero. *)
let nonzero_products c a b =
  let nonzero x =
    let m =
      List.fold_left Nx.add (Nx.scalar D.Float64 0.)
        (List.map Nx.abs (parts64 x))
    in
    Nx.cast D.Float64 (Nx.not_equal_s m 0.)
  in
  Nx.to_array
    (P.eval ~by:"t"
       (contraction_op
          { c with sa = None; sb = None }
          ~acc:(D.Any D.Float64) D.Float64 (nonzero a) (nonzero b) None))

(* The least positive subnormal of a float accumulator's parts. *)
let subnormal (D.Any dt) =
  match dt with
  | D.Float64 | D.Complex128 -> Float.ldexp 1. (-1074)
  | _ -> Float.ldexp 1. (-149)

(* Integers give eager's bits. A float result of [K] contracted terms is within
   [γ(K + 1 + s) (|init| + Σ|a·b|)] of the exact one on each side, [s] its
   scaled operands, a complex result's part within [γ(K + 2)] of that part's
   magnitudes, and each of its roundings errs by half a subnormal more where it
   underflows. With an init, a zero result's sign follows the order of
   summation, which the bound leaves open: [+0] added last, or a [-0] init
   first; so does that of a result with a nonzero product, by whether tolk fuses
   it ([nonzero_products]). A NaN result's bits are unspecified. *)
let contract_check c =
  let (D.Any dt) = c.cdt in
  let a, b, init, e, l = contract_results dt c in
  match D.kind dt with
  | D.Float | D.Complex ->
      let k = numel c.ek in
      let s = Bool.to_int (c.sa <> None) + Bool.to_int (c.sb <> None) in
      let depth = s + if D.is D.Complex dt then k + 2 else k + 1 in
      let g = gamma (carrier_roundoff c.acc) depth in
      let under = Float.of_int (depth + 1) *. subnormal c.acc in
      let bits = part_bits dt and round = rounding_of dt in
      let products = nonzero_products c a b in
      let check (e, l) mags =
        let mags = Nx.to_array mags in
        Array.iteri
          (fun o e ->
            near
              ~zeros:(c.init = None && products.(o) = 0.)
              ~bits
              ~bound:((2. *. g *. mags.(o)) +. under +. round e l.(o))
              e l.(o))
          e
      in
      List.iter2 check
        (List.combine (parts dt e) (parts dt l))
        (magnitudes c a b init)
  | _ -> same_bits dt e l

let cover_contraction c =
  cover "a batch axis" (c.eb <> [||]);
  cover "two contracted axes" (Array.length c.ek = 2);
  cover "no contracted axis" (c.ek = [||]);
  cover "no contracted term" (numel c.ek = 0);
  cover "an init" (c.init <> None);
  cover "a transposed operand" (c.ta || c.tb)

(* A contraction of no term from an init, every run. *)
let empty dt =
  let c =
    {
      cdt = dt;
      acc = dt;
      eb = [||];
      ek = [| 0 |];
      fa = [| 2 |];
      fb = [||];
      ra = [| F 0; C 0 |];
      rb = [| C 0 |];
      ta = false;
      tb = false;
      araws = [||];
      braws = [||];
      init = None;
      sa = None;
      sb = None;
      seg = None;
    }
  in
  [ c; { c with init = Some (Array.make (raws_of dt 2) 0x80000000L) } ]

(* A contraction over two axes with an init, every run: [ones] are the raws
   of [1] of the dtype. *)
let two_axes dt ones =
  let c = List.hd (empty dt) in
  {
    c with
    ek = [| 2; 2 |];
    ra = [| F 0; C 0; C 1 |];
    rb = [| C 1; C 0 |];
    araws = Array.make 8 ones;
    braws = Array.make 4 ones;
    init = Some [| ones; ones |];
  }

(* A product of one free axis on each side by a batch axis, of no contracted
   axis, through a transposed [a], every run: [ones] as above. *)
let outer dt ones =
  let c = List.hd (empty dt) in
  {
    c with
    eb = [| 2 |];
    ek = [||];
    fa = [| 2 |];
    fb = [| 3 |];
    ra = [| F 0; B 0 |];
    rb = [| B 0; F 0 |];
    ta = true;
    araws = Array.make 4 ones;
    braws = Array.make 6 ones;
  }

let int_contractions =
  let accs d = [ d; D.Any D.Int64 ] in
  let wide = { (outer (D.Any D.Int32) 1L) with acc = D.Any D.Int64 } in
  prop ~count:12
    ~examples:(two_axes (D.Any D.Int32) 1L :: wide :: empty (D.Any D.Int32))
    "integer contractions wrap as eager's"
    (contraction (weigh 1 ints) accs)
    (fun c ->
      cover_contraction c;
      cover "a wider accumulator" (c.acc <> c.cdt);
      contract_check c)

let float_contractions =
  let accs (D.Any d as dt) =
    match d with
    | D.Float64 -> [ dt ]
    | D.Float32 -> [ dt; D.Any D.Float64 ]
    | _ -> [ D.Any D.Float32 ]
  in
  prop ~count:12
    ~examples:
      (two_axes (D.Any D.Float32) 0x3f800000L
      :: outer (D.Any D.Float32) 0x3f800000L
      :: empty (D.Any D.Float32))
    "float contractions hold their bound"
    (contraction (weigh 1 floats) accs)
    (fun c ->
      cover_contraction c;
      contract_check c)

let complex_contractions =
  prop ~count:6
    ~examples:(empty (D.Any D.Complex64))
    "complex contractions hold their bound"
    (contraction (weigh 1 complexes) (fun d -> [ d ]))
    (fun c ->
      cover "an init" (c.init <> None);
      contract_check c)

(* Scaled contractions: a code scale of a NaN code and a [float16] value scale
   on blocks of two, in [float64]. *)
let scaled_example =
  let c = two_axes (D.Any D.Float32) 0x3f800000L in
  {
    c with
    acc = D.Any D.Float64;
    sa = Some (Codes ([| 1; 1; 2 |], [| 127L; 255L |]));
    sb = Some (Values (D.Any D.Float16, [| 2; 1 |], [| 0x3c00L; 0xc000L |]));
  }

(* Elements of a narrow float, [ones] the raws of its [1], scaled by codes
   on both sides, as microscaling formats are, in [float32]. *)
let coded dt ones =
  let c = two_axes dt ones in
  {
    c with
    acc = D.Any D.Float32;
    sa = Some (Codes ([| 1; 1; 2 |], [| 127L; 126L |]));
    sb = Some (Codes ([| 2; 1 |], [| 127L; 128L |]));
  }

let scaled_contractions =
  let accs (D.Any d as dt) =
    match d with
    | D.Float64 -> [ dt ]
    | _ -> [ D.Any D.Float32; D.Any D.Float64 ]
  in
  let codes = function Some (Codes (_, r)) -> Array.to_list r | _ -> [] in
  let several = function
    | Some (Codes (b, _) | Values (_, b, _)) -> Array.exists (( < ) 1) b
    | None -> false
  in
  prop ~count:12
    ~examples:
      [
        scaled_example;
        coded (D.Any D.Float8_e4m3fn) 0x38L;
        coded (D.Any D.Float4_e2m1fn) 0x2L;
      ]
    "scaled float contractions hold their bound"
    (contraction ~extras:Scaled
       (weigh 1
          D.
            [
              Any Float64;
              Any Float32;
              Any Float16;
              Any Bfloat16;
              Any Float8_e4m3fn;
              Any Float4_e2m1fn;
            ])
       accs)
    (fun c ->
      cover "a float8_e4m3fn operand" (c.cdt = D.Any D.Float8_e4m3fn);
      cover "a float4_e2m1fn operand" (c.cdt = D.Any D.Float4_e2m1fn);
      let is_code = function Some (Codes _) -> true | _ -> false in
      let is_value = function Some (Values _) -> true | _ -> false in
      cover "a code scale" (is_code c.sa || is_code c.sb);
      cover "a value scale" (is_value c.sa || is_value c.sb);
      cover "both operands scaled" (c.sa <> None && c.sb <> None);
      cover "a NaN code" (List.mem 255L (codes c.sa @ codes c.sb));
      cover "a scale of several elements" (several c.sa || several c.sb);
      cover "a float64 accumulator" (c.acc = D.Any D.Float64);
      contract_check c)

(* Segmented contractions: rows whose group [1] is empty and of which two are
   in no group; rows by groups of matrices; a pair cut into three indices
   and one by an int64 extreme; decreasing offsets. *)
let segmented_examples =
  let c = List.hd (empty (D.Any D.Float32)) in
  let ones n = Array.make n 0x3f800000L in
  let rows =
    {
      c with
      ek = [| 2 |];
      fa = [| 3 |];
      fb = [| 2 |];
      ra = [| F 0; C 0 |];
      rb = [| C 0; F 0 |];
      araws = ones 6;
      braws = ones 4;
      seg = Some (Rows (0, 0, [| 0L; 1L; 1L |]));
    }
  in
  let pair o =
    {
      c with
      ek = [| 4 |];
      fa = [| 2 |];
      ra = [| F 0; C 0 |];
      rb = [| C 0 |];
      araws = ones 8;
      braws = ones 4;
      seg = Some (Contracting (0, 2, o));
    }
  in
  (* x[2, 3, K]·W[K, 2, 2] from [-0], its second free axis the rows', row
     [0] in no group, W's slices of [1] and [2]. *)
  let grouped =
    {
      rows with
      fa = [| 2; 3 |];
      fb = [| 2; 2 |];
      ra = [| F 0; F 1; C 0 |];
      rb = [| C 0; F 0; F 1 |];
      araws = ones 12;
      braws =
        Array.init 8 (fun i ->
            if i / 2 mod 2 = 0 then 0x3f800000L else 0x40000000L);
      init = Some (Array.make 12 0x80000000L);
      seg = Some (Rows (1, 0, [| 1L; 2L; 3L |]));
    }
  in
  [ rows; grouped; pair [| 0L; 3L; Int64.max_int |]; pair [| 0L; 3L; 2L |] ]

let segmented_contractions =
  let accs (D.Any d as dt) =
    match D.kind d with
    | D.Float when not (D.equal d D.Float64) -> [ D.Any D.Float32 ]
    | D.Signed | D.Unsigned -> [ dt; D.Any D.Int64 ]
    | _ -> [ dt ]
  in
  prop ~count:16 ~examples:segmented_examples
    "segmented contractions give eager's integers and hold their bound"
    (contraction ~extras:Segmented
       (weigh 2 D.[ Any Float32; Any Float64 ]
       @ weigh 1 D.[ Any Int32; Any Uint8; Any Complex64; Any Float16 ])
       accs)
    (fun c ->
      let g, n = groups c in
      let os = batches c in
      let in_group o r = member o ~g r <> None in
      let indices = List.init n Fun.id and groups = List.init g Fun.id in
      let count o e =
        List.length (List.filter (fun r -> member o ~g r = Some e) indices)
      in
      let any f = List.exists f os in
      let extreme x = x = Int64.max_int || x = Int64.min_int in
      let rows = match c.seg with Some (Rows _) -> true | _ -> false in
      cover "rows" rows;
      cover "rows beside another free axis of a"
        (rows && Array.length c.fa > 1);
      cover "rows by groups beside a free axis of b"
        (rows && Array.length c.fb > 1);
      cover "a contracting pair"
        (match c.seg with Some (Contracting _) -> true | _ -> false);
      cover "an index in no group"
        (any (fun o -> List.exists (fun r -> not (in_group o r)) indices));
      cover "an empty group"
        (any (fun o -> List.exists (fun e -> count o e = 0) groups));
      cover "a group of several indices"
        (any (fun o -> List.exists (fun e -> count o e > 1) groups));
      cover "an int64 extreme" (any (Array.exists extreme));
      cover "decreasing offsets"
        (any (fun o -> List.exists (fun i -> o.(i + 1) < o.(i)) groups));
      contract_check c)

(* Cases *)

let f32 = Nx.create D.Float32

(* A max-pooling window: the padded load's fill enters the windows it
   overlaps. *)
let pooled m fill =
  let x = f32 [| 6 |] [| 1.; Float.nan; -0.; 0.; 3.; Float.neg_infinity |] in
  let pad =
    {
      K.lo = Iarray.of_list [ 1 ];
      hi = Iarray.of_list [ 1 ];
      interior = Iarray.of_list [ 0 ];
      windows =
        Iarray.of_list
          [ { Nx_array.Move.axis = 0; size = 3; step = 2; dilation = 1 } ];
    }
  in
  let op () =
    fst
      (P.eval ~by:"t"
         (P.Reduce
            {
              layout = Nx_array.Layout.contiguous [| 3; 3 |];
              axes = Iarray.of_list [ 1 ];
              prog = identity D.Float32;
              reductions = [ Monoid (m, 0, D.Float32) ];
              loads = Iarray.of_list [ P.Padded { x; fill; pad } ];
            }))
  in
  same_values D.Float32 (Nx.to_array (op ())) (Lowered.run op)

(* Three reductions of one output in one loop. *)
let several () =
  let x = Nx.create D.Int16 [| 2; 3 |] [| 32767; 1; -2; -32768; 0; 5 |] in
  let op k () =
    let s, (mx, (mn, ())) =
      P.eval ~by:"t"
        (P.Reduce
           {
             layout = Nx_array.Layout.contiguous [| 2; 3 |];
             axes = Iarray.of_list [ 1 ];
             prog = identity D.Int16;
             reductions =
               [
                 Monoid (Sum, 0, D.Int16);
                 Monoid (Max, 0, D.Int16);
                 Monoid (Min, 0, D.Int16);
               ];
             loads = Iarray.of_list [ P.Plain x ];
           })
    in
    List.nth [ s; mx; mn ] k
  in
  List.iter
    (fun k -> equal (array int) (Nx.to_array (op k ())) (Lowered.run (op k)))
    [ 0; 1; 2 ]

(* 16-bit products past [int]'s range: [65535 * 65535] wraps. *)
let uint16_products ~scan () =
  let x =
    Nx.create D.Uint16 [| 2; 3 |] [| 65535; 65535; 65535; 2; 65535; 32768 |]
  in
  let op () =
    if scan then P.eval ~by:"t" (scan_op Prod 1 x None)
    else fst (P.eval ~by:"t" (reduce_op Prod [ 1 ] x))
  in
  equal (array int) (Nx.to_array (op ())) (Lowered.run op)

let int16_contraction () =
  let a = Nx.create D.Int16 [| 4 |] [| -1; -32768; 32767; -32768 |] in
  let b = Nx.create D.Int16 [| 4 |] [| -1; -32768; -32768; 32767 |] in
  let batch = Iarray.of_list [] and contracting = Iarray.of_list [ (0, 0) ] in
  let op () =
    P.eval ~by:"t"
      (P.Contract
         {
           batch;
           contracting;
           acc = D.Any D.Int16;
           out = D.Int16;
           a;
           b;
           init = None;
           a_scale = None;
           b_scale = None;
           segments = None;
         })
  in
  equal (array int) (Nx.to_array (op ())) (Lowered.run op)

(* A [2; 0] by [0; 3] contraction has no products: each output is its init
   as the accumulator holds it, rounded into [dt], bit for bit. *)
let no_products (type v s) (dt : (v, s) D.t) acc raws () =
  let a = value dt [| 2; 0 |] [||] and b = value dt [| 0; 3 |] [||] in
  let init = value dt [| 2; 3 |] raws in
  let batch = Iarray.of_list [] and contracting = Iarray.of_list [ (1, 0) ] in
  let op () =
    P.eval ~by:"t"
      (P.Contract
         {
           batch;
           contracting;
           acc;
           out = dt;
           a;
           b;
           init = Some init;
           a_scale = None;
           b_scale = None;
           segments = None;
         })
  in
  same_bits dt (Nx.to_array init) (Nx.to_array (op ()));
  same_bits dt (Nx.to_array (op ())) (Lowered.run op)

(* -0, +0, +inf, -inf, a NaN with a payload and 1.5. *)
let f32_inits =
  [| 0x80000000L; 0L; 0x7f800000L; 0xff800000L; 0x7fc00001L; 0x3fc00000L |]

let f16_inits = [| 0x8000L; 0L; 0x7c00L; 0xfc00L; 0x7e01L; 0x3e00L |]

let f64_inits =
  [|
    Int64.min_int;
    0L;
    0x7ff0000000000000L;
    0xfff0000000000000L;
    0x7ff8000000000001L;
    0x3ff8000000000000L;
  |]

(* A contraction with a scale or segments is not lowered. *)
let int_scaled () =
  let a = Nx.create D.Int32 [| 2; 4 |] (Array.init 8 Int32.of_int) in
  let b = Nx.create D.Int32 [| 4; 2 |] (Array.init 8 Int32.of_int) in
  let codes = Nx.create D.Uint8 [| 2; 2 |] [| 127; 128; 126; 127 |] in
  let op () =
    P.eval ~by:"t"
      (P.Contract
         {
           batch = Iarray.of_list [];
           contracting = Iarray.of_list [ (1, 0) ];
           acc = D.Any D.Int32;
           out = D.Int32;
           a;
           b;
           init = None;
           a_scale = None;
           b_scale = Some (P.Exponent codes);
           segments = None;
         })
  in
  raises_match
    (Exn.invalid_arg ~substring:"a contraction scaled in int32 is not lowered")
    (fun () -> Lowered.run op)

(* A complex product, and a complex128 extreme's scan, are not lowered. *)
let refused_complex ~scan m () =
  let x =
    Nx.create D.Complex128 [| 2 |]
      [| { Complex.re = 1.; im = 2. }; { re = 3.; im = -1. } |]
  in
  let op () =
    if scan then P.eval ~by:"t" (scan_op m 0 x None)
    else fst (P.eval ~by:"t" (reduce_op m [ 0 ] x))
  in
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Lowered.run op)

(* A boolean is its byte: the bytes a reduction or a scan of booleans leaves are
   eager's. *)
let bool_bytes ~scan m () =
  let x =
    Nx.bitcast D.Bool
      (Nx.create D.Uint8 [| 3; 3 |] [| 2; 0; 3; 0; 0; 0; 4; 5; 255 |])
  in
  let op () =
    let y =
      if scan then P.eval ~by:"t" (scan_op m 1 x None)
      else fst (P.eval ~by:"t" (reduce_op m [ 1 ] x))
    in
    Nx.bitcast D.Uint8 y
  in
  equal (array int) (Nx.to_array (op ())) (Lowered.run op)

(* A float32 running variance of [0, 4e19, 0, 0]: the squared deviations
   overflow from the second term on, which gives eager's [+∞] variances. *)
let moments_overflow () =
  let x = Nx.create D.Float32 [| 4 |] [| 0.; 4e19; 0.; 0. |] in
  let c = { spread_example with shape = [| 4 |]; axes = [ 0 ] } in
  let op () = snd (by ~scan:true (Moments (0, D.Float32)) c x ()) in
  let inf = Float.infinity in
  equal (array float_exact) [| 0.; inf; inf; inf |] (Lowered.run op)

let cases =
  group "cases"
    [
      test "booleans' any keeps eager's bytes" (bool_bytes ~scan:false Max);
      test "booleans' all keeps eager's bytes" (bool_bytes ~scan:false Min);
      test "booleans' running any keeps eager's bytes"
        (bool_bytes ~scan:true Max);
      test "a max-pooling window gives eager's values" (fun () ->
          pooled Max Float.neg_infinity);
      test "a min-pooling window gives eager's values" (fun () ->
          pooled Min Float.infinity);
      test "several reductions of one loop" several;
      test "a uint16 product wraps past int" (uint16_products ~scan:false);
      test "a uint16 running product wraps past int"
        (uint16_products ~scan:true);
      test "an int16 contraction wraps at the extremes" int16_contraction;
      test "a float32 contraction of no products is its init"
        (no_products D.Float32 (D.Any D.Float32) f32_inits);
      test "a float64 contraction of no products is its init"
        (no_products D.Float64 (D.Any D.Float64) f64_inits);
      test "a float16 contraction of no products in float32 is its init"
        (no_products D.Float16 (D.Any D.Float32) f16_inits);
      test "a contraction scaled in int32 is not lowered" int_scaled;
      test "a complex product is refused" (refused_complex ~scan:false Prod);
      test "a complex running product is refused"
        (refused_complex ~scan:true Prod);
      test "a complex128 running maximum is refused"
        (refused_complex ~scan:true Max);
      test "a running variance that overflows is +∞" moments_overflow;
    ]

let () =
  exit
    (run "Lower_reduce"
       [
         group "reductions"
           [ extremes; int_folds; float_sums; float_products; complex_sums ];
         group "scans" [ extreme_scans; int_scans; float_scans; complex_scans ];
         group "logsumexp, moments and arg"
           [
             logsumexps;
             logsumexp_scans;
             moment_reductions;
             moment_scans;
             args;
             arg_scans;
           ];
         group "contractions"
           [
             int_contractions;
             float_contractions;
             complex_contractions;
             scaled_contractions;
             segmented_contractions;
           ];
         cases;
       ])
