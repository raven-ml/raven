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
  | D.Complex -> float_raw ?special ~spread ~subnormal D.Float64
  | _ -> int_raw dt

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

(* Forms

   A case's form is all of it but its elements: its dtype, monoid, shape, view
   and axes, and whether a scan has a carry. A law draws its cases' forms from a
   few, which its examples take too, so that a run lowers and compiles each form
   once and draws the elements of each case anew. *)

module T = struct
  let f64 = D.Any D.Float64
  let f32 = D.Any D.Float32
  let f16 = D.Any D.Float16
  let bf16 = D.Any D.Bfloat16
  let f8e4 = D.Any D.Float8_e4m3fn
  let f8e5 = D.Any D.Float8_e5m2
  let f4 = D.Any D.Float4_e2m1fn
  let i64 = D.Any D.Int64
  let u64 = D.Any D.Uint64
  let i32 = D.Any D.Int32
  let u32 = D.Any D.Uint32
  let i16 = D.Any D.Int16
  let u16 = D.Any D.Uint16
  let i8 = D.Any D.Int8
  let u8 = D.Any D.Uint8
  let i4 = D.Any D.Int4
  let u4 = D.Any D.Uint4
  let bool = D.Any D.Bool
  let bit = D.Any D.Bit
  let c64 = D.Any D.Complex64
  let c128 = D.Any D.Complex128
end

let form ?(view = Plain) ?(carry = false) dt (m : K.monoid) shape axes =
  {
    dt;
    m;
    shape;
    view;
    axes;
    raws = [||];
    carry = (if carry then Some [||] else None);
  }

(* [elements ~special ~spread ~subnormal ~long f] draws the elements of a case
   of the form [f], as {!float_raw} draws a float, and its carry's where [f] has
   one. Where [long], float terms are zeros, infinities and NaNs about once in
   1600, so that most rows stay numbers through several chunks. *)
let elements ?special ?(spread = 4) ?(subnormal = 1) ?(long = false) f =
  let open Gen in
  let term =
    if not long then raw ?special ~spread ~subnormal f.dt
    else
      frequency
        [
          (512, raw ~special:0 ~spread ~subnormal f.dt);
          (1, raw ~spread ~subnormal f.dt);
        ]
  in
  let* raws =
    array ~size:(constant (raws_of f.dt (numel (base f.view f.shape)))) term
  in
  let+ carry =
    match f.carry with
    | None -> constant None
    | Some _ ->
        let c = K.carrier f.dt in
        let n = raws_of c (numel (carried f.shape (List.hd f.axes))) in
        map Option.some (array ~size:(constant n) (raw ~spread ~subnormal c))
  in
  { f with raws; carry }

(* [case forms] draws a case of a form of [forms], each as likely. *)
let case ?spread ?subnormal ?long forms =
  Gen.with_pp pp_case
    (Gen.bind (among forms) (elements ?spread ?subnormal ?long))

let operand (type v s) (dt : (v, s) D.t) c : (v, s, Nx.host) Nx.t =
  viewed c.view c.shape (value dt (base c.view c.shape) c.raws)

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
      at = None;
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
      at = None;
    }

(* [monoid_op dt ~scan c] is [c]'s reduction, or scan where [scan]. *)
let monoid_op (type v s) (dt : (v, s) D.t) ~scan c () : (v, s, Nx.host) Nx.t =
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
    P.eval ~by:"t" (scan_op c.m axis x carry)
  else fst (P.eval ~by:"t" (reduce_op c.m c.axes x))

(* [results c] is the eager and the lowered elements of [c]'s reduction or
   scan. *)
let results (type v s) (dt : (v, s) D.t) ~scan c : v array * v array =
  let op = monoid_op dt ~scan c in
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

(* Each part of the eager and the lowered results of a float sum of a
   reduction's [n] terms is within [γ(β(n)) Σ|x|] of the exact sum,
   [β(n) = ⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉], the kernel contract's bound, that
   of a scan's terms up to a result, the carry's included, within
   [γ(n - 1) Σ|x|], and a product within a relative [γ(n - 1)] of the exact
   product: [Σ|x|] is eager's
   float64 sum of the terms' magnitudes. A zero sum keeps its sign where it is
   the carrier's, or where every term is a zero: the rounding into a narrower
   destination gives a zero of the carrier sum's sign, which within the bound
   can be either. *)
let bounded ~scan c =
  let (D.Any dt) = c.dt in
  let narrow =
    match dt with
    | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
    | D.Float4_e2m1fn ->
        true
    | _ -> false
  in
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
  let beta n = ((5 * min n 256) + 3) / 4 + ((n + 127) / 128) in
  let u = carrier_roundoff c.dt in
  let g = gamma u (n - 1) in
  let gs = if scan then g else gamma u (beta n) in
  let bits = part_bits dt and round = rounding_of dt in
  let check (e, l) (x, carry) =
    let mags = magnitudes x carry in
    Array.iteri
      (fun o e ->
        let l = l.(o) in
        let bound =
          match c.m with
          | Prod -> 2.01 *. g *. Float.max (Float.abs e) (Float.abs l)
          | _ -> 2. *. gs *. mags.(o)
        in
        let zeros = c.m <> Sum || (not narrow) || mags.(o) = 0. in
        near ~zeros ~bits ~bound:(bound +. round e l) e l)
      e
  in
  List.iter2 check
    (List.combine (parts dt e) (parts dt l))
    (List.combine xs carries)

let is_bool (D.Any dt) = D.is D.Boolean dt
let is_complex (D.Any dt) = D.is D.Complex dt
let reduced_empty c = List.exists (fun a -> c.shape.(a) = 0) c.axes
let cover_monoids c ms = List.iter (fun m -> cover (monoid_name m) (c.m = m)) ms

(* Examples every run takes, one on each form a law draws. A form of five rows
   of four of float32 or float16, along its rows, takes five rows: NaNs of both
   signs and kinds; infinities of both signs before a signalling NaN; zeros of
   both signs; all [-0]; and infinities of both signs, a NaN made from numbers
   by a sum or a product. A scan's carries are then [1], a signalling NaN, [+0],
   [-0] and [-∞]. Another form takes the raws [0], [1], [2], ..., and a carry of
   ones. *)

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

let examples fs =
  let example f =
    let n = raws_of f.dt (numel (base f.view f.shape)) in
    let ones () =
      let axis = List.hd f.axes in
      Array.make (raws_of (K.carrier f.dt) (numel (carried f.shape axis))) 1L
    in
    match specials f.dt with
    | Some (raws, carry)
      when f.shape = [| 5; 4 |] && f.axes = [ 1 ] && f.view = Plain ->
        { f with raws; carry = Option.map (fun _ -> carry) f.carry }
    | _ ->
        {
          f with
          raws = Array.init n Int64.of_int;
          carry = Option.map (fun _ -> ones ()) f.carry;
        }
  in
  List.map example fs

(* The shape of the special rows. *)
let rows = [| 5; 4 |]

(* [samples ~scan dts ms] is the examples every run takes on each dtype of [dts]
   by each monoid of [ms]: one element of ones, from a carry of ones where
   [scan]; a zero-size operand; where it reduces, the raws [0], [1], ... of a
   transposed operand along both its axes; and for float32, float16 and
   complex64 the special rows above. *)
let samples ?(scan = false) dts ms =
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
          [ ex dt m rows [ 1 ] raws (if scan then Some carry else None) ]
    in
    let carry =
      if scan then Some (Array.make (raws_of (K.carrier dt) 1) 1L) else None
    in
    let across =
      if scan then []
      else
        let raws = Array.init (raws_of dt 6) Int64.of_int in
        [ { (ex dt m [| 3; 2 |] [ 0; 1 ] raws None) with view = Transposed } ]
    in
    ex dt m [| 1 |] [ 0 ] (Array.make (raws_of dt 1) 1L) carry
    :: ex dt m zero [ 1 ] [||] None
    :: (across @ special)
  in
  List.concat_map (fun dt -> List.concat_map (each dt) ms) dts

(* The form of the case [c]. *)
let blank c = { c with raws = [||]; carry = Option.map (fun _ -> [||]) c.carry }

(* [law ~on ~graph ~count forms name f] holds [f] on the examples [on], on an
   example of each of [forms], and on [count] cases of the forms of both, the
   kernels of each example's [graph] made together first. *)
let law ?spread ?subnormal ?long ?(on = []) ~graph ~count forms name f =
  let examples = on @ examples forms in
  prop ~count ~examples name
    (case ?spread ?subnormal ?long (List.map blank on @ forms))
    (Lowered.primed graph examples f)

(* The graph of [c]'s reduction or scan, but a refused reduction's. *)
let monoid_graph ~scan c =
  if (not scan) && refused c then []
  else
    let (D.Any dt) = c.dt in
    [ Lowered.One (monoid_op dt ~scan c) ]

(* Reductions

   Each law takes its samples, and forms that reach each dtype the law draws,
   each view, a zero-size and a one-element operand, several axes and an empty
   reduced extent. *)

let extremes =
  law ~graph:(monoid_graph ~scan:false)
    ~on:(samples T.[ f32; f16; bool; i8; c64 ] [ Max; Min ])
    ~count:12
    T.
      [
        form ~view:Broadcast i8 Min [| 2; 3; 4 |] [ 0; 2 ];
        form bit Min [| 1 |] [ 0 ];
        form ~view:Broadcast bit Max [| 2; 3 |] [ 1 ];
        form f64 Max [| 0; 2 |] [ 1 ];
        form ~view:Flipped bf16 Min [| 3; 4 |] [ 0 ];
        form c128 Max [| 2; 3 |] [ 0; 1 ];
        form ~view:Transposed f4 Max [| 2; 3 |] [ 0 ];
        form u64 Min [| 4 |] [ 0 ];
        form ~view:Flipped f8e4 Max [| 2; 4 |] [ 1 ];
        form f8e5 Min [| 3; 1; 2 |] [ 0; 2 ];
        form ~view:Transposed i64 Max [| 4; 2 |] [ 1 ];
        form i32 Min [| 2; 3 |] [ 0 ];
        form ~view:Broadcast u32 Max [| 3; 4 |] [ 0; 1 ];
        form i16 Min [| 0; 3 |] [ 1 ];
        form ~view:Flipped u16 Max [| 3; 2; 2 |] [ 2 ];
        form u8 Min [| 1; 4 |] [ 0; 1 ];
        form ~view:Transposed i4 Max [| 4; 3 |] [ 0; 1 ];
        form u4 Min [| 2; 2; 2 |] [ 1 ];
      ]
    "extremes give eager's values"
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
  law ~graph:(monoid_graph ~scan:false)
    ~on:(samples T.[ i8 ] [ Sum; Prod ])
    ~count:6
    T.
      [
        form ~view:Broadcast i64 Sum [| 2; 3; 4 |] [ 0; 2 ];
        form u32 Prod [| 2; 0 |] [ 1 ];
        form u4 Sum [| 1 |] [ 0 ];
        form ~view:Flipped i16 Prod [| 4; 3 |] [ 0 ];
        form u64 Sum [| 3 |] [ 0 ];
        form i4 Prod [| 2; 3 |] [ 1 ];
        form ~view:Transposed i32 Sum [| 2; 4 |] [ 0 ];
        form ~view:Broadcast u8 Prod [| 3; 3 |] [ 0; 1 ];
        form u16 Sum [| 2; 3; 2 |] [ 0; 2 ];
      ]
    "integer sums and products wrap as eager's"
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "an empty reduced extent" (reduced_empty c);
      cover_shapes c;
      exact ~scan:false c)

let float_sums =
  law ~graph:(monoid_graph ~scan:false)
    ~on:(samples T.[ f32; f16 ] [ Sum ])
    ~count:10
    T.
      [
        form ~view:Transposed f64 Sum [| 3; 2 |] [ 0; 1 ];
        form ~view:Broadcast bf16 Sum [| 2; 3; 4 |] [ 0; 2 ];
        form f8e4 Sum [| 2; 0 |] [ 1 ];
        form f4 Sum [| 1 |] [ 0 ];
        form ~view:Flipped f8e5 Sum [| 4; 3 |] [ 0 ];
      ]
    "float sums hold their bound"
    (fun c ->
      cover_floats c;
      cover "an empty reduced extent" (reduced_empty c);
      cover "a narrow float"
        (not (List.mem c.dt D.[ Any Float32; Any Float64 ]));
      bounded ~scan:false c)

(* Products of numbers of one binade: no overflow or underflow. *)
let float_products =
  law ~graph:(monoid_graph ~scan:false) ~spread:0 ~subnormal:0
    ~on:(samples T.[ f32 ] [ Prod ])
    ~count:6
    T.
      [
        form ~view:Transposed f64 Prod [| 3; 2 |] [ 0; 1 ];
        form f16 Prod [| 2; 0 |] [ 1 ];
        form ~view:Broadcast bf16 Prod [| 2; 3; 4 |] [ 0; 2 ];
        form f8e5 Prod [| 3 |] [ 0 ];
        form ~view:Flipped f8e4 Prod [| 3; 2 |] [ 1 ];
        form f4 Prod [| 2; 2 |] [ 0; 1 ];
      ]
    "float products hold their bound"
    (fun c ->
      cover "a NaN term"
        (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
      cover "an empty reduced extent" (reduced_empty c);
      bounded ~scan:false c)

let complex_sums =
  law ~graph:(monoid_graph ~scan:false)
    ~on:(samples T.[ c64 ] [ Sum ])
    ~count:4
    T.
      [
        form ~view:Transposed c128 Sum [| 3; 2 |] [ 0; 1 ];
        form c128 Sum [| 1 |] [ 0 ];
        form ~view:Broadcast c64 Sum [| 2; 3 |] [ 0 ];
        form ~view:Flipped c128 Sum [| 2; 3; 2 |] [ 1; 2 ];
      ]
    "complex sums add their parts"
    (fun c ->
      cover "complex64" (c.dt = D.Any D.Complex64);
      bounded ~scan:false c)

(* Scans *)

let extreme_scans =
  law ~graph:(monoid_graph ~scan:true)
    ~on:(samples ~scan:true T.[ f32; f16; bool; c64 ] [ Max; Min ])
    ~count:6
    T.
      [
        form ~view:Transposed bool Max [| 3; 2 |] [ 0 ];
        form ~view:Broadcast ~carry:true i8 Min [| 2; 3; 4 |] [ 2 ];
        form ~carry:true bit Max [| 1 |] [ 0 ];
        form ~view:Flipped bit Min [| 3; 2 |] [ 0 ];
        form f64 Min [| 2; 0 |] [ 1 ];
        form ~view:Flipped u16 Max [| 4; 3 |] [ 0 ];
        form ~view:Flipped ~carry:true bf16 Max [| 3; 3 |] [ 1 ];
        form ~carry:true f8e4 Min [| 2; 4 |] [ 1 ];
        form ~view:Transposed f8e5 Max [| 3; 2 |] [ 1 ];
        form ~view:Broadcast ~carry:true f4 Min [| 2; 4 |] [ 1 ];
        form ~carry:true i64 Max [| 5 |] [ 0 ];
        form ~view:Transposed u64 Min [| 2; 3 |] [ 0 ];
        form ~view:Flipped i32 Max [| 3; 2 |] [ 1 ];
        form ~carry:true u32 Min [| 4; 2 |] [ 0 ];
        form ~view:Broadcast ~carry:true i16 Max [| 3; 2; 2 |] [ 1 ];
        form u8 Min [| 3; 3 |] [ 1 ];
        form ~carry:true i4 Max [| 1; 5 |] [ 1 ];
        form ~view:Transposed u4 Min [| 2; 2; 3 |] [ 2 ];
      ]
    "extreme scans give eager's values"
    (fun c ->
      cover_monoids c [ Max; Min ];
      cover "a carry" (c.carry <> None);
      cover "a NaN term"
        (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
      cover "a boolean" (is_bool c.dt);
      cover "a complex number" (is_complex c.dt);
      exact ~scan:true c)

let int_scans =
  law ~graph:(monoid_graph ~scan:true)
    ~on:(samples ~scan:true T.[ i8 ] [ Sum; Prod ])
    ~count:2
    T.
      [
        form ~carry:true i8 Sum [| 2; 3 |] [ 1 ];
        form ~view:Transposed i8 Prod [| 3; 2 |] [ 0 ];
        form ~view:Broadcast ~carry:true u64 Sum [| 4; 3 |] [ 1 ];
        form ~carry:true i4 Prod [| 1 |] [ 0 ];
        form i32 Sum [| 2; 0 |] [ 1 ];
        form ~view:Flipped ~carry:true u16 Prod [| 3; 4 |] [ 1 ];
        form ~view:Transposed ~carry:true i64 Prod [| 3; 3 |] [ 1 ];
        form u32 Sum [| 5 |] [ 0 ];
        form ~view:Flipped ~carry:true i16 Prod [| 2; 2; 3 |] [ 2 ];
        form ~view:Broadcast u8 Sum [| 2; 4 |] [ 0 ];
        form ~carry:true u4 Sum [| 3; 2 |] [ 0 ];
      ]
    "integer running sums and products wrap as eager's"
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "a carry" (c.carry <> None);
      exact ~scan:true c)

let float_scans =
  law ~graph:(monoid_graph ~scan:true) ~spread:0 ~subnormal:0
    ~on:(samples ~scan:true T.[ f32; f16 ] [ Sum; Prod ])
    ~count:7
    T.
      [
        form ~view:Transposed f64 Sum [| 3; 2 |] [ 0 ];
        form ~view:Broadcast ~carry:true bf16 Prod [| 2; 3; 4 |] [ 2 ];
        form ~carry:true f8e4 Sum [| 1 |] [ 0 ];
        form ~view:Flipped f4 Prod [| 4; 3 |] [ 0 ];
        form ~carry:true f8e5 Sum [| 3; 3 |] [ 1 ];
      ]
    "float running sums and products hold their bound"
    (fun c ->
      cover_monoids c [ Sum; Prod ];
      cover "a carry" (c.carry <> None);
      bounded ~scan:true c)

let complex_scans =
  law ~graph:(monoid_graph ~scan:true)
    ~on:(samples ~scan:true T.[ c64 ] [ Sum ])
    ~count:2
    T.
      [
        form ~view:Transposed c128 Sum [| 3; 2 |] [ 0 ];
        form ~view:Flipped ~carry:true c128 Sum [| 2; 3 |] [ 1 ];
        form ~view:Broadcast c64 Sum [| 2; 3 |] [ 0 ];
      ]
    "complex running sums add their parts"
    (fun c ->
      cover "a carry" (c.carry <> None);
      bounded ~scan:true c)

(* Long axes. Examples every run takes, one for each monoid the law draws:
   float64 sums of 600 terms, one row of numbers, one with a NaN, one with both
   infinities; an int8 sum of 40 extremes from a carry at an extreme; an int32
   product of 40 terms that wraps; a float64 product of 600 terms near 1; a
   float32 maximum of 600 terms with a NaN late in the row, flipped; a float64
   minimum of 600 terms with a [-∞]; a complex128 sum of 600 numbers;
   float4_e2m1fn and float8_e4m3fn sums from a carry of [-2⁻³⁰] of 16 zeros,
   then [x], [-x], ..., whose zero results round a carrier sum of either sign,
   as [x] absorbs the carry or not. *)
let long_examples =
  let cancelling dt x neg_x =
    {
      dt;
      m = Sum;
      shape = [| 40 |];
      view = Plain;
      axes = [ 0 ];
      raws =
        Array.init 40 (fun i ->
            if i < 16 then 0L else if i mod 2 = 0 then x else neg_x);
      carry = Some [| 0xb0800000L |];
    }
  in
  let f64 xs = Array.map Int64.bits_of_float xs in
  let f32 xs = Array.map (fun x -> Int64.of_int32 (Int32.bits_of_float x)) xs in
  let n = 600 in
  let numbers i = Float.of_int ((i mod 7) - 3) +. 0.25 in
  let row k i =
    match (k, i) with
    | 1, 100 -> Float.nan
    | 2, 20 -> Float.infinity
    | 2, 300 -> Float.neg_infinity
    | _ -> numbers i
  in
  let extremes = Array.init 80 (fun i -> if i mod 3 = 0 then 127L else -128L) in
  [
    {
      dt = D.Any D.Float64;
      m = Sum;
      shape = [| 3; n |];
      view = Plain;
      axes = [ 1 ];
      raws = f64 (Array.init (3 * n) (fun i -> row (i / n) (i mod n)));
      carry = None;
    };
    {
      dt = D.Any D.Int8;
      m = Sum;
      shape = [| 2; 40 |];
      view = Plain;
      axes = [ 1 ];
      raws = extremes;
      carry = Some [| 127L; -128L |];
    };
    {
      dt = D.Any D.Int32;
      m = Prod;
      shape = [| 40 |];
      view = Plain;
      axes = [ 0 ];
      raws = Array.init 40 (fun i -> Int64.of_int ((i mod 5) - 7));
      carry = None;
    };
    {
      dt = D.Any D.Float64;
      m = Prod;
      shape = [| n |];
      view = Transposed;
      axes = [ 0 ];
      raws = f64 (Array.init n (fun i -> 1. +. (numbers i /. 64.)));
      carry = None;
    };
    {
      dt = D.Any D.Float32;
      m = Max;
      shape = [| n |];
      view = Flipped;
      axes = [ 0 ];
      raws =
        f32 (Array.init n (fun i -> if i = 50 then Float.nan else numbers i));
      carry = None;
    };
    {
      dt = D.Any D.Float64;
      m = Min;
      shape = [| n |];
      view = Plain;
      axes = [ 0 ];
      raws =
        f64
          (Array.init n (fun i ->
               if i = 400 then Float.neg_infinity else numbers i));
      carry = None;
    };
    {
      dt = D.Any D.Complex128;
      m = Sum;
      shape = [| n |];
      view = Plain;
      axes = [ 0 ];
      raws = f64 (Array.init (2 * n) numbers);
      carry = None;
    };
    cancelling (D.Any D.Float4_e2m1fn) 0x7L 0xfL;
    cancelling (D.Any D.Float8_e4m3fn) 0x78L 0xf8L;
  ]

(* Beside the examples' forms, a form of each other dtype the law draws, of axes
   of 33 to 96 and 513 to 640 terms at several residues of the chunks' 16: sums
   and products, a maximum or a minimum of every float, integer and boolean, and
   complex64's sum. *)
let long_forms =
  T.
    [
      form ~view:Transposed ~carry:true f32 Sum [| 3; 75 |] [ 1 ];
      form ~carry:true f16 Sum [| 513 |] [ 0 ];
      form ~view:Flipped bf16 Sum [| 2; 58 |] [ 1 ];
      form ~view:Broadcast f8e5 Sum [| 2; 567 |] [ 1 ];
      form ~carry:true i64 Sum [| 640 |] [ 0 ];
      form ~view:Flipped u64 Prod [| 47 |] [ 0 ];
      form ~view:Transposed u32 Sum [| 96; 2 |] [ 0 ];
      form ~carry:true i16 Prod [| 2; 33 |] [ 1 ];
      form ~view:Broadcast u16 Sum [| 2; 631; 2 |] [ 1 ];
      form u8 Prod [| 64 |] [ 0 ];
      form ~view:Flipped ~carry:true i4 Sum [| 545 |] [ 0 ];
      form ~view:Transposed u4 Prod [| 2; 81 |] [ 1 ];
      form ~carry:true i64 Max [| 2; 89 |] [ 1 ];
      form ~view:Broadcast u8 Min [| 2; 522 |] [ 1 ];
      form ~view:Transposed bool Max [| 71; 2 |] [ 0 ];
      form ~carry:true bool Min [| 614 |] [ 0 ];
      form ~view:Flipped f16 Max [| 2; 37 |] [ 1 ];
      form ~carry:true bf16 Min [| 530 |] [ 0 ];
      form ~view:Transposed f8e4 Max [| 66; 2 |] [ 0 ];
      form ~view:Broadcast f8e5 Min [| 2; 601 |] [ 1 ];
      form ~carry:true f4 Max [| 2; 50 |] [ 1 ];
      form ~view:Flipped u64 Min [| 517 |] [ 0 ];
      form ~carry:true i32 Max [| 2; 79 |] [ 1 ];
      form ~view:Transposed u32 Min [| 548; 2 |] [ 0 ];
      form i16 Max [| 2; 35 |] [ 1 ];
      form ~view:Broadcast ~carry:true u16 Min [| 2; 92 |] [ 1 ];
      form ~view:Flipped i8 Min [| 625 |] [ 0 ];
      form ~carry:true i4 Max [| 2; 41 |] [ 1 ];
      form ~view:Transposed u4 Min [| 587; 2 |] [ 0 ];
      form ~view:Flipped ~carry:true c64 Sum [| 2; 52 |] [ 1 ];
    ]

(* A case of a form of [forms], each as likely: a float sum's or product's terms
   of one binade, as the bound of a long sum or product needs. *)
let long_case forms =
  let elements f =
    let (D.Any dt) = f.dt in
    match (D.kind dt, f.m) with
    | D.Float, (Sum | Prod) -> elements ~spread:0 ~subnormal:0 ~long:true f
    | _ -> elements ~long:true f
  in
  Gen.with_pp pp_case (Gen.bind (among forms) elements)

let long_scans =
  let forms = List.map blank long_examples @ long_forms in
  let examples = long_examples @ examples long_forms in
  prop ~count:16 ~examples "running scans along long axes give eager's values"
    (long_case forms)
    (Lowered.primed (monoid_graph ~scan:true) examples (fun c ->
         let n = c.shape.(List.hd c.axes) in
         cover_monoids c [ Sum; Prod; Max; Min ];
         cover "an axis in chunks" (n > 32 && n <= 512);
         cover "an axis in chunks of chunks" (n > 512);
         cover "a carry" (c.carry <> None);
         cover "a strided or broadcast view" (c.view <> Plain);
         cover "a NaN term"
           (float_raws c (fun dt r -> float_class dt r = Nan) <> []);
         cover "an infinity"
           (float_raws c (fun dt r -> float_class dt r = Inf) <> []);
         cover "a complex number" (is_complex c.dt);
         let (D.Any dt) = c.dt in
         match (D.kind dt, c.m) with
         | (D.Float | D.Complex), (Sum | Prod) -> bounded ~scan:true c
         | _ -> exact ~scan:true c))

(* Logsumexp, Moments and Arg

   Arg gives eager's values and positions. Logsumexp and Moments, into the
   operand's dtype, hold the bounds Lower_reduce states, plus the rounding
   into a destination narrower than the carrier, against a float64 reference:
   OCaml's [Float] functions on the terms, the special values from
   {!Nx_kernel.type-reduction}. The reference is in error by the reduction's
   bounds at float64's unit roundoff, with glibc's [exp] and [log] within an
   ulp, and by its rounding into the destination; for float64 the two errors
   add. *)

(* [E] and [L], the bounds in ulps of the compiled [Exp] and [Log] at float32
   and float64: nx's, since both compile to nx's bits. *)
let exp_ulps = 2.
let log_ulps = 2.
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
           at = None;
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
              at = None;
            }))

(* The graphs of an operation's two results, each its own: a user who reads one
   computes that one alone. *)
let each_result op =
  Lowered.[ One (fun () -> fst (op ())); One (fun () -> snd (op ())) ]

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
  (* The references rounded into [dt], as the results are. Into float32 that
     rounding errs by up to [u|e|], which [rounding] counts only for a
     narrower [dt]. *)
  let rounded es =
    let es = Nx.create D.Float64 [| Array.length es |] es in
    Nx.to_array (Nx.cast D.Float64 (Nx.cast dt es))
  in
  let into = match dt with D.Float32 -> u | _ -> 0. in
  let close n e l ~ours ~reference =
    let reference = reference +. (into *. Float.abs e) in
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

(* The graph of [c]'s logsumexp, or its moments. *)
let float_graph ~scan which c =
  let (D.Any dt) = c.dt in
  match D.kind dt with
  | D.Float -> (
      let x = operand dt c in
      match which with
      | `Logsumexp -> [ Lowered.One (by ~scan (Logsumexp (0, dt)) c x) ]
      | `Moments -> each_result (by ~scan (Moments (0, dt)) c x))
  | _ -> []

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

(* The terms [16 + 2^-18] and [0x1.27428cp+4], whose logsumexp lies within
   [0.006] ulp of the midpoint of two float32s: a result within its bound may
   round to either, and the reference to the other. *)
let midpoint_row =
  {
    spread_example with
    shape = [| 1; 2 |];
    raws = [| 0x41800002L; 0x4193a146L |];
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

(* The examples of a law of floats: those above, and float32's, float16's and
   float64's samples, without a carry. *)
let float_examples ~scan =
  spread_example :: midpoint_row :: sine_row T.f32 1500 :: sine_row T.f16 300
  :: List.map
       (fun c -> { c with carry = None })
       (samples ~scan T.[ f32; f16; f64 ] [ Sum ])

(* The other floats' forms, of each view, and a row of hundreds to thousands of
   terms of each float the examples' rows are not of. *)
let float_forms ~scan =
  T.
    [
      form ~view:Broadcast bf16 Sum [| 2; 3 |] [ 1 ];
      form ~view:Flipped f8e4 Sum [| 3; 4 |] (if scan then [ 1 ] else [ 0; 1 ]);
      form f8e5 Sum [| 2; 2; 3 |] [ 2 ];
      form ~view:Transposed f4 Sum [| 4; 2 |] [ 0 ];
      form f64 Sum [| 1; 1800 |] [ 1 ];
      form bf16 Sum [| 1; 700 |] [ 1 ];
      form f8e4 Sum [| 1; 130 |] [ 1 ];
      form f8e5 Sum [| 1; 450 |] [ 1 ];
      form f4 Sum [| 1; 1000 |] [ 1 ];
    ]

let float_prop ~scan which name =
  let extra = float_forms ~scan in
  let examples = float_examples ~scan @ examples extra in
  let forms = List.map blank (float_examples ~scan) @ extra in
  (* A row of hundreds to thousands of finite terms within [2^±spread] reaches
     the bounds' growth with the number of terms. *)
  let long_row =
    let open Gen in
    let* f = among (List.filter (fun f -> numel f.shape > 100) forms) in
    let* spread = among [ 0; 2; 30 ] in
    elements ~special:0 ~spread ~subnormal:0 f
  in
  prop ~count:10 ~examples name
    (Gen.frequency
       [ (3, case ~spread:30 forms); (1, Gen.with_pp pp_case long_row) ])
    (Lowered.primed (float_graph ~scan which) examples (fun c ->
         cover "an empty axis" (List.exists (fun a -> c.shape.(a) = 0) c.axes);
         floating ~scan which c))

let logsumexps =
  float_prop ~scan:false `Logsumexp "a logsumexp holds its bound"

let logsumexp_scans =
  float_prop ~scan:true `Logsumexp "a running logsumexp holds its bound"

let moment_reductions =
  float_prop ~scan:false `Moments "moments hold their bounds"

let moment_scans =
  float_prop ~scan:true `Moments "running moments hold their bounds"

(* [arg_prop ~scan ~count forms name] is an arg's law on its samples, without a
   carry, and on [forms]. *)
let arg_prop ~scan ~count forms name =
  let on =
    List.map
      (fun c -> { c with carry = None })
      (samples ~scan T.[ f32; f16; bool; i8; c64 ] [ Max; Min ])
  in
  let graph c =
    let (D.Any dt) = c.dt in
    if (not scan) && refused c then []
    else each_result (by ~scan (Arg (extreme c, 0, dt)) c (operand dt c))
  in
  law ~on ~graph ~count forms name (fun c ->
      assume (scan || not (refused c));
      cover_monoids c [ Max; Min ];
      cover_floats c;
      if not scan then cover_shapes c;
      cover "a boolean" (is_bool c.dt);
      cover "a complex number" (is_complex c.dt);
      cover "an integer"
        (not (is_float c.dt || is_bool c.dt || is_complex c.dt));
      arg_law ~scan c)

let args =
  arg_prop ~scan:false ~count:2
    T.
      [
        form ~view:Broadcast i8 Max [| 2; 3; 4 |] [ 0; 2 ];
        form bit Max [| 1 |] [ 0 ];
        form ~view:Flipped bit Min [| 2; 3 |] [ 1 ];
        form f64 Min [| 0; 2 |] [ 1 ];
        form c128 Min [| 2; 3 |] [ 0; 1 ];
        form ~view:Flipped u8 Max [| 4; 3 |] [ 0 ];
        form ~view:Transposed bf16 Max [| 3; 2 |] [ 1 ];
        form f8e4 Min [| 2; 3 |] [ 0; 1 ];
        form ~view:Broadcast f8e5 Max [| 2; 4 |] [ 1 ];
        form ~view:Flipped f4 Min [| 3; 3 |] [ 0 ];
        form i64 Max [| 4 |] [ 0 ];
        form ~view:Transposed u64 Min [| 2; 3 |] [ 0 ];
        form ~view:Broadcast i32 Max [| 3; 2; 2 |] [ 0; 2 ];
        form u32 Min [| 1; 3 |] [ 1 ];
        form ~view:Flipped i16 Max [| 2; 4 |] [ 0; 1 ];
        form u16 Min [| 3; 0 |] [ 0 ];
        form ~view:Transposed i4 Max [| 2; 2; 2 |] [ 1 ];
        form u4 Min [| 5 |] [ 0 ];
      ]
    "an arg gives eager's values and positions"

let arg_scans =
  arg_prop ~scan:true ~count:2
    T.
      [
        form ~view:Transposed bool Min [| 3; 2 |] [ 0 ];
        form ~view:Broadcast i8 Max [| 2; 3; 4 |] [ 2 ];
        form bit Max [| 1 |] [ 0 ];
        form ~view:Flipped bit Min [| 3; 2 |] [ 1 ];
        form f64 Min [| 2; 0 |] [ 1 ];
        form ~view:Flipped u8 Min [| 4; 3 |] [ 0 ];
        form ~view:Transposed bf16 Max [| 3; 2 |] [ 1 ];
        form f8e4 Min [| 2; 3 |] [ 1 ];
        form ~view:Broadcast f8e5 Max [| 2; 4 |] [ 1 ];
        form ~view:Flipped f4 Min [| 3; 3 |] [ 0 ];
        form i64 Max [| 4 |] [ 0 ];
        form ~view:Transposed u64 Min [| 2; 3 |] [ 0 ];
        form ~view:Broadcast i32 Max [| 3; 2; 2 |] [ 2 ];
        form u32 Min [| 1; 3 |] [ 1 ];
        form ~view:Flipped i16 Max [| 2; 4 |] [ 1 ];
        form u16 Min [| 3; 2 |] [ 0 ];
        form ~view:Transposed i4 Max [| 2; 2; 2 |] [ 1 ];
        form u4 Min [| 5 |] [ 0 ];
      ]
    "a running arg gives eager's values and positions"

(* Contractions

   Each operand's axes are batch, free or contracted, in any order, read through
   a transposed view or not. An operand has at most one free axis, three for
   [Rows]; the result's free axes are [a]'s, then [b]'s, in axis order. A
   scale's blocks divide each of its operand's extents; codes stay below
   [151] or are [255], so that no partial sum overflows. Ids are drawn
   around the groups, int64's extremes included. *)

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

let ids_of = function Rows (_, _, o) | Contracting (_, _, o) -> o

(* [member o ~g r] is the group of index [r] by the ids [o], as
   {!Nx_kernel.type-segments} states. *)
let member o ~g r =
  let id = o.(r) in
  if id >= 0L && id < Int64.of_int g then Some (Int64.to_int id) else None

(* The ids of each batch index. *)
let batches c =
  let _, n = groups c in
  match c.seg with
  | None -> []
  | Some s ->
      let o = ids_of s in
      List.init (numel c.eb) (fun i -> Array.sub o (i * n) n)

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

(* Contraction forms

   A contraction's form is all of it but its elements: dtypes, extents, roles,
   views, whether it has an init, its scales' kinds and blocks, and its
   segments' axes and groups. A law draws its cases' forms from a few, which its
   examples take too. *)

let cform ?acc ?(eb = [||]) ?(ta = false) ?(tb = false) ?(init = false) ?sa ?sb
    ?seg cdt ~ek ~fa ~fb ra rb =
  {
    cdt;
    acc = Option.value acc ~default:cdt;
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
    init = (if init then Some [||] else None);
    sa;
    sb;
    seg;
  }

(* The form of the contraction [c]. *)
let shape_of c =
  let scale = function
    | Codes (b, _) -> Codes (b, [||])
    | Values (d, b, _) -> Values (d, b, [||])
  in
  let seg = function
    | Rows (i, j, _) -> Rows (i, j, [||])
    | Contracting (p, g, _) -> Contracting (p, g, [||])
  in
  {
    c with
    araws = [||];
    braws = [||];
    init = Option.map (fun _ -> [||]) c.init;
    sa = Option.map scale c.sa;
    sb = Option.map scale c.sb;
    seg = Option.map seg c.seg;
  }

(* [contraction_data ~crowd ~drop c] draws the elements of a case of the form
   [c]: its operands', its init's, its scales' codes or values and its segments'
   ids. Codes stay below [151] or are [255], so that no partial sum overflows.
   Ids are drawn around the groups, int64's extremes included. Where [crowd],
   they crowd their rows into group 0 in three draws of four, so that a group
   passes a block's rows; where [drop], the last group is left empty in half the
   draws. *)
let contraction_data ?(crowd = false) ?(drop = false) c =
  let open Gen in
  let raws s =
    array
      ~size:(constant (raws_of c.cdt (numel s)))
      (raw ~spread:2 ~subnormal:1 c.cdt)
  in
  let* araws = raws (ashape c c.fa c.ra) in
  let* braws = raws (ashape c c.fb c.rb) in
  let* init =
    match c.init with
    | None -> constant None
    | Some _ -> map Option.some (raws (out_shape c))
  in
  let code =
    frequency
      [
        (1, constant 0L);
        (1, constant 255L);
        (6, map Int64.of_int (int_range 100 150));
      ]
  in
  let scale = function
    | None -> constant None
    | Some (Codes (b, _)) ->
        map
          (fun r -> Some (Codes (b, r)))
          (array ~size:(constant (numel b)) code)
    | Some (Values (d, b, _)) ->
        map
          (fun r -> Some (Values (d, b, r)))
          (array ~size:(constant (numel b)) (raw ~spread:2 ~subnormal:1 d))
  in
  let* sa = scale c.sa in
  let* sb = scale c.sb in
  let* crowded =
    if crowd then frequency [ (1, constant false); (3, constant true) ]
    else constant false
  in
  let* empty = if drop then bool else constant false in
  let ids g n =
    let held =
      if g = 0 then [] else [ (6, map Int64.of_int (int_range 0 (g - 1))) ]
    in
    let entry =
      frequency
        (held
        @ [
            (3, map Int64.of_int (int_range (-1) g));
            (1, among [ Int64.min_int; Int64.max_int ]);
          ])
    in
    let entry =
      if crowded then frequency [ (19, constant 0L); (1, entry) ] else entry
    in
    array ~size:(constant (numel c.eb * n)) entry
  in
  let+ seg =
    match c.seg with
    | None -> constant None
    | Some (Rows (i, j, _)) ->
        let last = Int64.of_int (c.fb.(j) - 1) in
        let drop o =
          if empty then Array.map (fun x -> if x = last then -1L else x) o
          else o
        in
        map (fun o -> Some (Rows (i, j, drop o))) (ids c.fb.(j) c.fa.(i))
    | Some (Contracting (p, g, _)) ->
        map (fun o -> Some (Contracting (p, g, o))) (ids g c.ek.(p))
  in
  { c with araws; braws; init; sa; sb; seg }

(* The raws of [n] ones of the dtype [d]. *)
let ones (D.Any dt as d) n =
  let one =
    match dt with
    | D.Float64 | D.Complex128 | D.Complex64 -> Int64.bits_of_float 1.
    | D.Float32 -> 0x3f800000L
    | D.Float16 -> 0x3c00L
    | D.Bfloat16 -> 0x3f80L
    | D.Float8_e4m3fn -> 0x38L
    | D.Float8_e5m2 -> 0x3cL
    | D.Float4_e2m1fn -> 0x2L
    | _ -> 1L
  in
  if D.is D.Complex dt then
    Array.init (raws_of d n) (fun i -> if i mod 2 = 0 then one else 0L)
  else Array.make n one

(* The example of the form [c] of ones, of scales of one, and of the ids [id b
   r] of row or index [r] at batch index [b]. *)
let ones_on ?(id = fun _ _ -> 0L) c =
  let n s = numel s in
  let segment n = Array.init (numel c.eb * n) (fun e -> id (e / n) (e mod n)) in
  let scale = function
    | Codes (b, _) -> Codes (b, Array.make (n b) 127L)
    | Values (d, b, _) -> Values (d, b, ones d (n b))
  in
  {
    c with
    araws = ones c.cdt (n (ashape c c.fa c.ra));
    braws = ones c.cdt (n (ashape c c.fb c.rb));
    init = Option.map (fun _ -> ones c.cdt (n (out_shape c))) c.init;
    sa = Option.map scale c.sa;
    sb = Option.map scale c.sb;
    seg =
      Option.map
        (function
          | Rows (i, j, _) -> Rows (i, j, segment c.fa.(i))
          | Contracting (p, g, _) -> Contracting (p, g, segment c.ek.(p)))
        c.seg;
  }

(* [contraction forms] draws a case of a form of [forms], each as likely. *)
let contraction ?crowd ?drop forms =
  Gen.with_pp pp_contraction
    (Gen.bind (among forms) (contraction_data ?crowd ?drop))

(* [contractions ~count ~examples ~forms name f] holds [f] on [examples], on an
   example of ones on each of [forms], and on [count] cases of [forms] and the
   examples' forms. *)
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
  let _, n = groups c in
  let pos r x = Option.get (Array.find_index (( = ) x) r) in
  match c.seg with
  | Some (Rows (i, j, o)) ->
      let ids = Nx.create D.Int64 (Array.append c.eb [| n |]) o in
      Rows { ids; a_axis = pos c.ra (F i); b_axis = pos c.rb (F j) }
  | Some (Contracting (p, g, o)) ->
      let ids = Nx.create D.Int64 (Array.append c.eb [| n |]) o in
      Contracting { ids; groups = g; pair = p }
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
      sum = false;
    }

(* [c]'s operands, its init, and its contraction of them. *)
let contraction_of (type v s) (dt : (v, s) D.t) c =
  let a = contracted dt (ashape c c.fa c.ra) c.ta c.araws in
  let b = contracted dt (ashape c c.fb c.rb) c.tb c.braws in
  let init = Option.map (value dt (out_shape c)) c.init in
  let op () = P.eval ~by:"t" (contraction_op c ~acc:c.acc dt a b init) in
  (a, b, init, op)

let contract_results (type v s) (dt : (v, s) D.t) c =
  let a, b, init, op = contraction_of dt c in
  (a, b, init, Nx.to_array (op ()), Lowered.run op)

let contraction_graph c =
  let (D.Any dt) = c.cdt in
  let _, _, _, op = contraction_of dt c in
  [ Lowered.One op ]

let contractions ?crowd ?drop ~count ~examples ?(forms = []) name f =
  let drawn = List.sort_uniq compare (List.map shape_of examples) @ forms in
  let examples = examples @ List.map (fun f -> ones_on f) forms in
  prop ~count ~examples name
    (contraction ?crowd ?drop drawn)
    (Lowered.primed contraction_graph examples f)

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
   [γ(β(K) + s) (|init| + Σ|a·b|)] of the exact one on each side, [s] its
   scaled operands and [β(K) = ⌈5 min(K, 256) / 4⌉ + ⌈K / 128⌉] the kernel
   contract's, a complex result's part within [γ(β(K))] of that part's
   magnitudes compiled and [γ(β(2K))] eager, as a kernel library sums a part's
   [2K] terms, and each of its roundings errs by a subnormal step more where
   it underflows. With an init, a zero result's sign follows the order of
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
      let complex = D.is D.Complex dt in
      let beta n = ((5 * min n 256) + 3) / 4 + ((n + 127) / 128) in
      let compiled = s + beta k in
      let eager = s + beta (if complex then 2 * k else k) in
      let u = carrier_roundoff c.acc in
      let g = gamma u compiled +. gamma u eager in
      let under = 2. *. Float.of_int (eager + 1) *. subnormal c.acc in
      let bits = part_bits dt and round = rounding_of dt in
      let products = nonzero_products c a b in
      let check (e, l) mags =
        let mags = Nx.to_array mags in
        Array.iteri
          (fun o e ->
            near
              ~zeros:(c.init = None && products.(o) = 0.)
              ~bits
              ~bound:((g *. mags.(o)) +. under +. round e l.(o))
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
  let wide = { (outer (D.Any D.Int32) 1L) with acc = D.Any D.Int64 } in
  contractions ~count:12
    ~examples:(two_axes (D.Any D.Int32) 1L :: wide :: empty (D.Any D.Int32))
    ~forms:
      T.
        [
          cform i8 ~eb:[| 2 |] ~ek:[| 3 |] ~fa:[| 2 |] ~fb:[| 2 |] ~ta:true
            ~tb:true ~init:true [| B 0; C 0; F 0 |] [| F 0; B 0; C 0 |];
          cform u64 ~ek:[| 2; 3 |] ~fa:[||] ~fb:[| 3 |] ~tb:true [| C 1; C 0 |]
            [| C 0; F 0; C 1 |];
          cform i64 ~ek:[| 2 |] ~fa:[| 3 |] ~fb:[| 2 |] [| F 0; C 0 |]
            [| C 0; F 0 |];
          cform u32 ~acc:i64 ~eb:[| 2 |] ~ek:[| 3 |] ~fa:[||] ~fb:[| 2 |]
            ~ta:true [| B 0; C 0 |] [| B 0; F 0; C 0 |];
          cform i16 ~ek:[| 4 |] ~fa:[| 1 |] ~fb:[| 3 |] ~init:true
            [| C 0; F 0 |] [| F 0; C 0 |];
          cform u16 ~acc:i64 ~ek:[| 2; 2 |] ~fa:[| 2 |] ~fb:[||] ~tb:true
            [| F 0; C 0; C 1 |] [| C 1; C 0 |];
          cform u8 ~ek:[| 3 |] ~fa:[| 2 |] ~fb:[| 2 |] ~ta:true [| C 0; F 0 |]
            [| C 0; F 0 |];
          cform i4 ~eb:[| 2 |] ~ek:[| 2 |] ~fa:[| 1 |] ~fb:[| 2 |] ~init:true
            [| B 0; F 0; C 0 |] [| B 0; C 0; F 0 |];
          cform u4 ~acc:i64 ~ek:[| 5 |] ~fa:[| 2 |] ~fb:[| 1 |] [| F 0; C 0 |]
            [| C 0; F 0 |];
        ]
    "integer contractions wrap as eager's"
    (fun c ->
      cover_contraction c;
      cover "a wider accumulator" (c.acc <> c.cdt);
      contract_check c)

let float_contractions =
  contractions ~count:12
    ~examples:
      (two_axes (D.Any D.Float32) 0x3f800000L
      :: outer (D.Any D.Float32) 0x3f800000L
      :: empty (D.Any D.Float32))
    ~forms:
      T.
        [
          cform f64 ~eb:[| 1 |] ~ek:[| 3 |] ~fa:[| 2 |] ~fb:[| 1 |] ~ta:true
            [| F 0; B 0; C 0 |] [| B 0; C 0; F 0 |];
          cform f16 ~acc:f32 ~ek:[| 3; 2 |] ~fa:[| 2 |] ~fb:[| 2 |] ~init:true
            [| C 0; F 0; C 1 |] [| C 1; F 0; C 0 |];
          cform bf16 ~acc:f32 ~ek:[| 2 |] ~fa:[| 3 |] ~fb:[||] ~ta:true
            [| F 0; C 0 |] [| C 0 |];
          cform f32 ~acc:f64 ~ek:[| 4 |] ~fa:[| 2 |] ~fb:[| 3 |] ~tb:true
            ~init:true [| C 0; F 0 |] [| F 0; C 0 |];
          cform f8e5 ~acc:f32 ~ek:[| 3 |] ~fa:[| 1 |] ~fb:[| 2 |] [| F 0; C 0 |]
            [| C 0; F 0 |];
          cform f4 ~acc:f32 ~eb:[| 2 |] ~ek:[| 2 |] ~fa:[| 2 |] ~fb:[| 2 |]
            [| B 0; F 0; C 0 |] [| C 0; B 0; F 0 |];
          cform f8e4 ~acc:f32 ~ek:[| 2 |] ~fa:[| 2 |] ~fb:[| 3 |] ~tb:true
            ~init:true [| F 0; C 0 |] [| F 0; C 0 |];
        ]
    "float contractions hold their bound"
    (fun c ->
      cover_contraction c;
      contract_check c)

let complex_contractions =
  contractions ~count:6
    ~examples:(empty (D.Any D.Complex64))
    ~forms:
      T.
        [
          cform c64 ~ek:[| 3 |] ~fa:[| 2 |] ~fb:[| 2 |] ~tb:true ~init:true
            [| F 0; C 0 |] [| C 0; F 0 |];
          cform c128 ~eb:[| 2 |] ~ek:[| 2 |] ~fa:[| 2 |] ~fb:[| 1 |] ~ta:true
            [| B 0; F 0; C 0 |] [| B 0; C 0; F 0 |];
        ]
    "complex contractions hold their bound"
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
  let codes = function Some (Codes (_, r)) -> Array.to_list r | _ -> [] in
  let several = function
    | Some (Codes (b, _) | Values (_, b, _)) -> Array.exists (( < ) 1) b
    | None -> false
  in
  contractions ~count:12
    ~examples:
      [
        scaled_example;
        coded (D.Any D.Float8_e4m3fn) 0x38L;
        coded (D.Any D.Float4_e2m1fn) 0x2L;
      ]
    ~forms:
      T.
        [
          cform f64 ~ek:[| 4 |] ~fa:[| 3 |] ~fb:[| 2 |]
            ~sa:(Values (f32, [| 3; 2 |], [||]))
            ~sb:(Codes ([| 2; 1 |], [||]))
            [| F 0; C 0 |] [| C 0; F 0 |];
          cform bf16 ~acc:f32 ~ek:[| 2 |] ~fa:[| 2 |] ~fb:[| 3 |]
            ~sb:(Codes ([| 1; 3 |], [||]))
            [| F 0; C 0 |] [| C 0; F 0 |];
          cform f16 ~acc:f64 ~eb:[| 2 |] ~ek:[| 2 |] ~fa:[| 1 |] ~fb:[||]
            ~sa:(Values (f64, [| 2; 1; 1 |], [||]))
            [| B 0; C 0; F 0 |] [| B 0; C 0 |];
        ]
    "scaled float contractions hold their bound"
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

(* Segmented contractions: rows whose group [1] is empty and one of which is in
   no group; rows by groups of matrices, one in none; a pair of one group and an
   int64 extreme; unsorted ids and an int64 extreme. *)
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
      seg = Some (Rows (0, 0, [| 0L; -1L; 0L |]));
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
      seg = Some (Rows (1, 0, [| -1L; 0L; 1L |]));
    }
  in
  [
    rows;
    grouped;
    pair [| 0L; 0L; 0L; Int64.max_int |];
    pair [| 1L; 0L; Int64.min_int; 0L |];
  ]

let segmented_contractions =
  contractions ~count:16 ~examples:segmented_examples
    ~forms:
      T.
        [
          cform f64 ~ek:[| 2 |] ~fa:[| 2; 3 |] ~fb:[| 3 |]
            ~seg:(Rows (1, 0, [||]))
            [| F 0; F 1; C 0 |] [| C 0; F 0 |];
          cform i32 ~acc:i64 ~ek:[| 5 |] ~fa:[| 2 |] ~fb:[||]
            ~seg:(Contracting (0, 3, [||]))
            [| F 0; C 0 |] [| C 0 |];
          cform c64 ~ek:[| 2 |] ~fa:[| 4 |] ~fb:[| 2 |] ~ta:true ~init:true
            ~seg:(Rows (0, 0, [||]))
            [| C 0; F 0 |] [| F 0; C 0 |];
          cform u8 ~ek:[| 3; 2 |] ~fa:[||] ~fb:[| 2 |]
            ~seg:(Contracting (1, 2, [||]))
            [| C 0; C 1 |] [| C 1; F 0; C 0 |];
          cform f16 ~acc:f32 ~eb:[| 2 |] ~ek:[| 3 |] ~fa:[| 3 |] ~fb:[| 2 |]
            ~seg:(Rows (0, 0, [||]))
            [| B 0; F 0; C 0 |] [| B 0; F 0; C 0 |];
        ]
    "segmented contractions give eager's integers and hold their bound"
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
      cover "unsorted ids"
        (any (fun o ->
             List.exists
               (fun i -> o.(i + 1) < o.(i))
               (List.init (max 0 (n - 1)) Fun.id)));
      contract_check c)

(* Contracting pairs long enough to be cut into blocks of [⌈n / g⌉] indices:
   groups of several blocks, empty groups and indices in no group. *)
(* 30 indices of x[2, 30]·y[30] in 3 groups, group [1] empty and indices
   [0] and [1] in none; with a second pair of 2, read transposed in y, and of
   none from an init of [-0], which the empty sum keeps. *)
let blocked_examples =
  let c = List.hd (empty (D.Any D.Float32)) in
  let ones n = Array.make n 0x3f800000L in
  let one =
    {
      c with
      ek = [| 30 |];
      fa = [| 2 |];
      ra = [| F 0; C 0 |];
      rb = [| C 0 |];
      araws = ones 60;
      braws = ones 30;
      seg =
        Some
          (Contracting
             ( 0,
               3,
               Array.init 30 (fun r ->
                   if r < 2 then -1L else if r mod 3 = 0 then 0L else 2L) ));
    }
  in
  let two =
    {
      one with
      ek = [| 30; 2 |];
      ra = [| F 0; C 0; C 1 |];
      rb = [| C 1; C 0 |];
      araws = ones 120;
      braws = ones 60;
    }
  in
  let none =
    {
      two with
      ek = [| 30; 0 |];
      araws = [||];
      braws = [||];
      init = Some (Array.make 6 0x80000000L);
    }
  in
  (* 400 indices in 3 groups, a quarter in none: past the host's budget of
     comparisons, the indices route by the running count's search. *)
  let long =
    {
      one with
      ek = [| 400 |];
      araws = ones 800;
      braws = ones 400;
      seg =
        Some
          (Contracting
             ( 0,
               3,
               Array.init 400 (fun r ->
                   if r mod 4 = 3 then -1L else Int64.of_int (r mod 3)) ));
    }
  in
  [ one; two; none; long ]

let blocked_contractions =
  contractions ~count:16 ~examples:blocked_examples
    ~forms:
      T.
        [
          cform f64 ~ek:[| 36 |] ~fa:[| 2 |] ~fb:[||]
            ~seg:(Contracting (0, 4, [||]))
            [| F 0; C 0 |] [| C 0 |];
          cform i32 ~acc:i64 ~ek:[| 24; 1 |] ~fa:[| 1 |] ~fb:[| 2 |] ~init:true
            ~seg:(Contracting (0, 3, [||]))
            [| C 1; F 0; C 0 |] [| C 0; C 1; F 0 |];
          cform u8 ~acc:i64 ~ek:[| 30 |] ~fa:[| 1 |] ~fb:[| 2 |]
            ~seg:(Contracting (0, 5, [||]))
            [| C 0; F 0 |] [| C 0; F 0 |];
          cform c64 ~ek:[| 28 |] ~fa:[| 2 |] ~fb:[| 1 |] ~tb:true
            ~seg:(Contracting (0, 2, [||]))
            [| F 0; C 0 |] [| F 0; C 0 |];
          cform f16 ~acc:f32 ~ek:[| 40 |] ~fa:[||] ~fb:[| 2 |] ~init:true
            ~seg:(Contracting (0, 4, [||]))
            [| C 0 |] [| C 0; F 0 |];
        ]
    "contracting pairs in blocks give eager's integers and hold their bound"
    (fun c ->
      let g, n = groups c in
      let os = batches c in
      let block = (n + g - 1) / g in
      let count o e =
        List.length
          (List.filter (fun r -> member o ~g r = Some e) (List.init n Fun.id))
      in
      let any f = List.exists f os in
      let groups = List.init g Fun.id in
      cover "a group of several blocks"
        (any (fun o -> List.exists (fun e -> count o e > block) groups));
      cover "an empty group"
        (any (fun o -> List.exists (fun e -> count o e = 0) groups));
      cover "an index in no group"
        (any (fun o -> List.fold_left (fun k e -> k + count o e) 0 groups < n));
      cover "a second contracting pair" (Array.length c.ek = 2);
      cover "no product" (numel c.ek = 0);
      contract_check c)

(* Rows of one group in one block at each of two batch indices: [a] of
   [2; 48] and [b] of [2; 1], each of [0.25]. Clang 19's straight-line
   vectorizer compiled the search of each slot's row for AVX2 into garbage
   rows, and five products read [0]. *)
let batched_block =
  let c = List.hd (empty (D.Any D.Float32)) in
  {
    c with
    eb = [| 2 |];
    ek = [||];
    fa = [| 48 |];
    fb = [| 1 |];
    ra = [| B 0; F 0 |];
    rb = [| B 0; F 0 |];
    araws = Array.make 96 0x3e800000L;
    braws = Array.make 2 0x3e800000L;
    seg = Some (Rows (0, 0, Array.make 96 0L));
  }

(* The elements of a group's slice of [Rows] over [g] groups, segmenting [b]'s
   first free axis: its contracted and other free extents. *)
let slice c g = numel c.ek * (numel c.fb / max 1 g)

(* Whether [Rows] of [n] rows over [g] groups, past 16 rows a group, in the
   host's blocks of 8 at each of [bs] batch indices route by comparisons:
   their [S n] comparisons a batch index, [S] slots, within the host's budget
   of [2^16]. *)
let compared ~bs ~n g =
  let m = min n g in
  let slots = (m + ((n - m) / 8)) * 8 in
  bs * slots * n <= 1 lsl 16

(* 300 rows over 257 groups, each row's slice of 257: the groups' first
   blocks pass the host's budget of [g²] comparisons, [2^16], and come from
   a scan. Every seventh row is in no group, the others spread over the
   groups, some empty. *)
let many_groups =
  let c = List.hd (empty (D.Any D.Float32)) in
  let g = 257 in
  {
    c with
    ek = [||];
    fa = [| 300 |];
    fb = [| g; g |];
    ra = [| F 0 |];
    rb = [| F 0; F 1 |];
    araws =
      Array.init 300 (fun r ->
          if r mod 2 = 0 then 0x3e800000L else 0xbf800000L);
    braws =
      Array.init (g * g) (fun i ->
          Int64.of_int (0x3f800000 + ((i mod 5) lsl 20)));
    seg =
      Some
        (Rows
           ( 0,
             0,
             Array.init 300 (fun r ->
                 if r mod 7 = 0 then -1L else Int64.of_int (r * 31 mod 200)) ));
  }

(* Forms of [Rows] past 16 rows a group, routed by comparisons at two batch
   indices and alone, and by the search; their examples' ids crowd a group past
   64 rows, leave one empty, which takes a block of padding alone, and leave
   rows in no group, with a part-full block at each batch index. *)
let blocked_rows_examples =
  let rows = Rows (0, 0, [||]) in
  T.
    [
      ones_on
        ~id:(fun b r ->
          if b = 0 then if r mod 10 = 9 then -1L else 0L
          else if r < 50 then 0L
          else -1L)
        (cform f32 ~eb:[| 2 |] ~ek:[| 8 |] ~fa:[| 100 |] ~fb:[| 1 |] ~seg:rows
           [| B 0; F 0; C 0 |] [| B 0; C 0; F 0 |]);
      ones_on
        ~id:(fun _ r -> if r mod 7 = 0 then -1L else 0L)
        (cform f64 ~ek:[| 6 |] ~fa:[| 280 |] ~fb:[| 2 |] ~seg:rows
           [| F 0; C 0 |] [| F 0; C 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 3))
        (cform i32 ~acc:i64 ~eb:[| 2 |] ~ek:[| 10; 2 |] ~fa:[| 72; 2 |]
           ~fb:[| 2 |] ~seg:rows
           [| B 0; F 0; C 0; F 1; C 1 |]
           [| C 1; B 0; F 0; C 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 2))
        (cform f16 ~acc:f32 ~ek:[| 9 |] ~fa:[| 60 |] ~fb:[| 2 |] ~ta:true
           ~init:true ~seg:rows [| F 0; C 0 |] [| C 0; F 0 |]);
      ones_on
        ~id:(fun _ r -> if r mod 5 = 4 then -1L else Int64.of_int (r mod 2))
        (cform u8 ~acc:i64 ~ek:[| 4 |] ~fa:[| 70 |] ~fb:[| 2 |] ~seg:rows
           [| F 0; C 0 |] [| F 0; C 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 2))
        (cform c64 ~ek:[| 3 |] ~fa:[| 50 |] ~fb:[| 2 |] ~ta:true ~seg:rows
           [| F 0; C 0 |] [| C 0; F 0 |]);
    ]

(* Rows of 24 a group or more, which the lowering routes into blocks of 64,
   of 8 on the host: groups past 64 rows, empty groups and rows in no
   group. *)
let blocked_rows =
  contractions ~crowd:true ~drop:true ~count:20 ~examples:blocked_rows_examples
    "rows in blocks give eager's integers and hold their bound" (fun c ->
      let g, n = groups c in
      let os = batches c in
      let count o e =
        List.length
          (List.filter (fun r -> member o ~g r = Some e) (List.init n Fun.id))
      in
      let any f = List.exists f os in
      let groups = List.init g Fun.id in
      let blocked = n >= 16 * g && g <= slice c g in
      cover "rows past 16 a group" blocked;
      cover "rows routed by comparisons"
        (blocked && compared ~bs:(numel c.eb) ~n g);
      cover "rows routed by the running count's search"
        (blocked && not (compared ~bs:(numel c.eb) ~n g));
      cover "a group past 64 rows"
        (any (fun o -> List.exists (fun e -> count o e > 64) groups));
      cover "an empty group"
        (any (fun o -> List.exists (fun e -> count o e = 0) groups));
      (* Rows in blocks take as many blocks as their groups could need, at
         least one a group: with a group empty, a block holds padding
         alone. *)
      cover "a block of padding alone"
        (n >= 16 * g
        && g <= slice c g
        && any (fun o -> List.exists (fun e -> count o e = 0) groups));
      cover "a row in no group"
        (any (fun o -> List.fold_left (fun k e -> k + count o e) 0 groups < n));
      cover "a batch axis" (Array.length c.eb > 0);
      cover "a part-full block at each of two batch indices"
        (numel c.eb = 2
        && List.for_all
             (fun o -> List.exists (fun e -> count o e mod 8 <> 0) groups)
             os);
      contract_check c)

(* Forms of [Rows] below 16 rows a group: fewer rows than groups, and 1 to 5 and
   5 to 16 a group, routed; their examples' ids leave a group empty and rows in
   no group. *)
let few_rows_examples =
  let rows = Rows (0, 0, [||]) in
  T.
    [
      ones_on
        ~id:(fun _ r -> [| 0L; 4L; -1L |].(r))
        (cform f32 ~ek:[| 6 |] ~fa:[| 3 |] ~fb:[| 5 |] ~seg:rows [| F 0; C 0 |]
           [| F 0; C 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 5))
        (cform f64 ~ek:[| 8 |] ~fa:[| 9 |] ~fb:[| 4 |] ~seg:rows [| F 0; C 0 |]
           [| C 0; F 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 2))
        (cform i32 ~acc:i64 ~ek:[| 7; 2 |] ~fa:[| 40 |] ~fb:[| 3 |] ~seg:rows
           [| F 0; C 0; C 1 |] [| C 1; F 0; C 0 |]);
      ones_on
        ~id:(fun _ r -> if r = 0 then 1L else 0L)
        (cform c64 ~ek:[| 6 |] ~fa:[| 2; 2 |] ~fb:[| 2; 3 |] ~seg:rows
           [| F 0; C 0; F 1 |] [| F 0; F 1; C 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 7))
        (cform f16 ~acc:f32 ~ek:[| 10 |] ~fa:[| 70 |] ~fb:[| 6 |] ~tb:true
           ~seg:rows [| F 0; C 0 |] [| F 0; C 0 |]);
      ones_on
        ~id:(fun _ r -> Int64.of_int (r mod 3))
        (cform u8 ~acc:i64 ~ek:[| 5 |] ~fa:[| 20 |] ~fb:[| 3 |] ~seg:rows
           [| F 0; C 0 |] [| C 0; F 0 |]);
    ]

(* Rows of fewer than 16 a group: below one a group each reads its own
   slice, above it the lowering routes them into blocks of 4 or 8 rows. *)
let few_rows =
  contractions ~drop:true ~count:27 ~examples:few_rows_examples
    "rows in small blocks give eager's integers and hold their bound" (fun c ->
      let g, n = groups c in
      let os = batches c in
      let count o e =
        List.length
          (List.filter (fun r -> member o ~g r = Some e) (List.init n Fun.id))
      in
      let any f = List.exists f os in
      let groups = List.init g Fun.id in
      let routed = g <= slice c g in
      cover "fewer rows than groups" (n < g);
      cover "1 to 5 rows a group, routed" (routed && g <= n && n < 5 * g);
      cover "5 to 16 rows a group, routed"
        (routed && 5 * g <= n && n < 16 * g);
      cover "an empty group"
        (g <= n && any (fun o -> List.exists (fun e -> count o e = 0) groups));
      cover "a row in no group"
        (any (fun o -> List.fold_left (fun k e -> k + count o e) 0 groups < n));
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
              at = None;
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
             at = None;
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
           sum = false;
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
           sum = false;
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
           sum = false;
         })
  in
  raises
    (Rune.Lower_arith.Unlowered
       "Rune.Lower_reduce: a contraction scaled in int32 is not lowered")
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
    (function Rune.Lower_arith.Unlowered _ -> true | _ -> false)
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

(* Work

   The estimated work and bytes loaded and stored of float32 contractions whose
   [n] indices are spread evenly over [g] groups, unsorted: [Rows] of x[n; k] by
   W[g; j; k], [Contracting] of x[n; k] by y[n; j] into [g; k; j], and the
   unsegmented contraction of x[n; k] by W[j; k] that both take as their unit,
   on an AVX2 host. The unit loads what its tile (Tolk.Heuristic.tile_rows)
   loads, and the processor's registers size the tile: with arm64's 32 registers
   or AVX-512's, the unit loads less than with AVX2's 16, while the segmented
   forms, whose rows share no tile, load the same. *)

let avx2 =
  lazy
    (Result.get_ok (Tolk.Helpers.Target.of_string "CPU:CLANG:x86_64,haswell"))

type form = Plain | Rows_of | Contracting_of

(* Where [scaled], [W] is float4_e2m1fn with a code scale per row of [k], as
   an MXFP4 weight enters, which a row reads by its group through the decode
   and the scale. *)
(* The float32 [Rows] product of x[n; k] by W[g; j; k], row [r] in group
   [r mod g]. *)
let rows_op ~n ~k ~j g =
  let ramp s = f32 s (Array.init (numel s) (fun i -> Float.of_int (i mod 7))) in
  let ids =
    Nx.create D.Int64 [| n |] (Array.init n (fun r -> Int64.of_int (r mod g)))
  in
  P.eval ~by:"t"
    (P.Contract
       {
         batch = Iarray.of_list [];
         contracting = Iarray.of_list [ (1, 2) ];
         acc = D.Any D.Float32;
         out = D.Float32;
         a = ramp [| n; k |];
         b = ramp [| g; j; k |];
         init = None;
         a_scale = None;
         b_scale = None;
         segments = Some (K.Rows { ids; a_axis = 0; b_axis = 0 });
         sum = false;
       })

let work ?(scaled = false) form ~n ~k ~j g =
  let ramp s = f32 s (Array.init (numel s) (fun i -> Float.of_int (i mod 7))) in
  let ids =
    Nx.create D.Int64 [| n |] (Array.init n (fun r -> Int64.of_int (r mod g)))
  in
  let run (type v s) (b : (v, s, Nx.host) Nx.t) contracting b_scale segments =
    Lowered.work ~target:(Lazy.force avx2) (fun () ->
        P.eval ~by:"t"
          (P.Contract
             {
               batch = Iarray.of_list [];
               contracting = Iarray.of_list contracting;
               acc = D.Any D.Float32;
               out = D.Float32;
               a = ramp [| n; k |];
               b;
               init = None;
               a_scale = None;
               b_scale;
               segments;
               sum = false;
             }))
  in
  (* [W] of shape [s], its last axis of [k]. *)
  let weight s contracting segments =
    if not scaled then run (ramp s) contracting None segments
    else
      let last = Array.length s - 1 in
      let codes = Array.mapi (fun i e -> if i = last then 1 else e) s in
      let scale = P.Exponent (Nx.full D.Uint8 codes 127) in
      run (Nx.cast D.Float4_e2m1fn (ramp s)) contracting (Some scale) segments
  in
  match form with
  | Plain -> weight [| j; k |] [ (1, 1) ] None
  | Rows_of ->
      weight [| g; j; k |] [ (1, 2) ]
        (Some (K.Rows { ids; a_axis = 0; b_axis = 0 }))
  | Contracting_of ->
      run (ramp [| n; j |]) [ (0, 0) ] None
        (Some (K.Contracting { ids; groups = g; pair = 0 }))

(* The estimated work of a float32 running sum of [n] terms, which grows
   linearly: its work per term at 65536 terms is at most twice a 32-term
   one's, where a sum of windows growing with the axis would be many times
   it. *)
let scan_work n =
  let x = f32 [| n |] (Array.init n (fun i -> Float.of_int (i mod 7))) in
  fst (Lowered.work (fun () -> P.eval ~by:"t" (scan_op Sum 0 x None)))

(* With fewer rows than groups, each row of [Rows] loads its own group's
   slice, so tolk cannot share a slice's loads, or [x]'s, between rows and
   columns as it does the unsegmented contraction's: a float32 one loads a
   few times its bytes, at most 5 (over 256 contracted, 1118720 against
   266240: the unsegmented one gathers each column of its slice once for 8
   rows), a float4 one, whose slice is an eighth of [x]'s bytes, more. A load
   of every group's slice would grow the bytes with [g], which grow no more
   than a quarter from 128 groups to 256. A scaled
   slice's decode is computed again in each row, so a scaled contraction's
   unit is [n] unsegmented contractions of one row by one slice. The bounds
   hold at any [k], 256 among them.

   With a row a group or more, the rows are routed into [S] slots of blocks
   of [c] rows of one group: the work is the unsegmented contraction of [S]
   rows, and the routing's, in [n g] and [S log n]. *)
let work_bounds =
  let n = 256 and j = 16 in
  let rows ~scaled ~k () =
    let n = 64 in
    let ops, lds = work ~scaled Plain ~n ~k ~j 1 in
    let at g =
      let o, l = work ~scaled Rows_of ~n ~k ~j g in
      let unit =
        if scaled then n * fst (work ~scaled Plain ~n:1 ~k ~j 1) else ops
      in
      at_most int ~msg:"operations" ~than:(2 * unit) o;
      if not scaled then at_most int ~msg:"bytes" ~than:(5 * lds) l;
      l
    in
    let l128 = at 128 in
    let l256 = at 256 in
    at_most int ~msg:"bytes at 256 groups" ~than:(l128 + (l128 / 4)) l256
  in
  let blocked ~k g () =
    (* An AVX2 host's block of rows: 4 below 5 rows a group on average, 8, its
       tile's rows, from 5. *)
    let c = if n < 5 * g then 4 else 8 in
    let slots = (g + ((n - g) / c)) * c in
    let unit = fst (work Plain ~n:slots ~k ~j 1) in
    let routing = 64 * ((n * g) + (slots * 9)) in
    let o, _ = work Rows_of ~n ~k ~j g in
    at_most int ~msg:"operations" ~than:((2 * unit) + routing) o
  in
  group "work"
    [
      test "a Rows contraction's work is one slice's at 128 and 256 groups"
        (rows ~scaled:false ~k:16);
      test "a scaled float4 Rows contraction's work is one slice's"
        (rows ~scaled:true ~k:16);
      test "a Rows contraction's work is one slice's over 256 contracted"
        (rows ~scaled:false ~k:256);
      test "a scaled float4 Rows contraction's work is one slice's over 256"
        (rows ~scaled:true ~k:256);
      test "a Rows contraction of fewer rows than groups is one kernel"
        (fun () ->
          equal int 1
            (Lowered.kernels (fun () -> rows_op ~n:24 ~k:16 ~j:16 32)));
      test "Rows of more groups than a slice's elements are one kernel"
        (fun () ->
          equal int 1
            (Lowered.kernels (fun () -> rows_op ~n:64 ~k:4 ~j:4 32)));
      test "a Rows contraction in blocks works as its slots' at 8 groups"
        (blocked ~k:16 8);
      (* 256 rows of one group make [S n = 2^16] comparisons, the host's
         budget, and route by them, the routed rows stored, since each of the
         product's columns reads them again; 257 pass it and take the running
         count's stages and the search. 300 rows over 256 groups sum the groups'
         first blocks in one reduce of [g² = 2^16] terms; over 257 they take
         a scan's stages. *)
      test "routed rows compare within the budget, past it they search"
        (fun () ->
          let kernels n g =
            Lowered.kernels (fun () -> rows_op ~n ~k:16 ~j:17 g)
          in
          equal (list int) [ 7; 9; 10; 12 ]
            [ kernels 256 1; kernels 257 1; kernels 300 256; kernels 300 257 ]);
      (* The routing does not depend on [k]: the work added by a deeper
         contraction is the products', which pad blocks of 8 rows at 8 rows
         a group to under twice the rows, [n + 7 g]. *)
      test "a Rows contraction of 8 rows a group pads under its rows"
        (fun () ->
          let g = 32 in
          let deeper w = fst (w ~k:256) - fst (w ~k:128) in
          let rows = deeper (fun ~k -> work Rows_of ~n ~k ~j g) in
          let plain = deeper (fun ~k -> work Plain ~n ~k ~j 1) in
          at_most int ~msg:"operations" ~than:(2 * plain) rows);
      test "a Rows contraction in blocks works as its slots' at 64 groups"
        (blocked ~k:256 64);
      test "a Contracting contraction of 8 groups works in blocks" (fun () ->
          let k = 16 in
          let ops, lds = work Plain ~n ~k ~j 1 in
          let o, l = work Contracting_of ~n ~k ~j 8 in
          at_most int ~msg:"operations" ~than:(4 * ops) o;
          at_most int ~msg:"bytes" ~than:(4 * lds) l);
      test
        "a running sum's work per term at most doubles from 32 to 65536 terms"
        (fun () ->
          at_most int ~than:(2 * 2048 * scan_work 32) (scan_work 65536));
    ]

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
      test "rows of one group in a block at each batch index give eager's values"
        (fun () -> contract_check batched_block);
      test "rows over 257 groups give eager's values" (fun () ->
          contract_check many_groups);
    ]

(* Sums of products

   A cotangent read by several products sums them, as the input of three
   projections takes [dY_q·W_qᵀ + dY_k·W_kᵀ + dY_v·W_vᵀ]: bfloat16 [T × d]
   cotangents by bfloat16 [d × d] weights, each product rounded to bfloat16 and
   the three added in bfloat16. sm_89's tensor core takes bfloat16 products, one
   a kernel, so there each product runs in a kernel of its own; the host has no
   core, and the three share one kernel there. Kept apart on the host, as
   sm_89's cores would keep them, they compute the one kernel's bits, from
   initial values too. *)

let draw dt i s = Nx.cast dt (Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s)
let pat = Nx.Pattern.v "t i, o i -> t o | i"

(* [Σ_i dY_i·W_iᵀ] over three [t × d] cotangents of [dt]. *)
let sum_of_products dt t d () =
  let dys = List.init 3 (fun i -> draw dt (30 + i) [| t; d |])
  and ws = List.init 3 (fun i -> draw dt (10 + i) [| d; d |]) in
  fun () ->
    match List.map2 (Nx.contract dt pat) dys ws with
    | [ q; k; v ] -> Nx.add (Nx.add q k) v
    | _ -> assert false

(* [Σ_i (B_i + dY_i·W_iᵀ)]: each product from an initial value, which it
   is stored past ({!Nx.contract}'s [init]). *)
let sum_of_products_from dt t d () =
  let dys = List.init 3 (fun i -> draw dt (30 + i) [| t; d |])
  and ws = List.init 3 (fun i -> draw dt (10 + i) [| d; d |])
  and bs = List.init 3 (fun i -> draw dt (50 + i) [| t; d |]) in
  fun () ->
    match
      List.map2
        (fun (dy, b) w -> Nx.contract ~init:b dt pat dy w)
        (List.combine dys bs) ws
    with
    | [ q; k; v ] -> Nx.add (Nx.add q k) v
    | _ -> assert false

let sm_89 = Result.get_ok (Tolk.Helpers.Target.of_string "CUDA:CUDA:sm_89")
let sm_89_cores = Tolk.Heuristic.cores (Tolk.Cstyle.without_compiler sm_89)

(* [apart f] checks that [f] kept apart as sm_89's cores keep it computes the
   bits of [f] in one kernel, both unoptimized: three kernels and one, so the
   two schedules compared differ. *)
let apart f () =
  let cores = [ ("CPU", sm_89_cores) ] in
  equal ~msg:"kernels apart" int 3 (Lowered.kernels ~cores f);
  equal ~msg:"kernels together" int 1 (Lowered.kernels f);
  Tolk.Setting.context
    [ Tolk.Setting.B (Tolk.Setting.noopt, true) ]
    (fun () ->
      let bits x = Array.map Int64.bits_of_float x in
      equal (array int64) (bits (Lowered.run f)) (bits (Lowered.run ~cores f)))

let sums_of_products =
  group "sums of products"
    [
      test "products kept apart on the host compute the one kernel's bits"
        (apart (sum_of_products Nx.bfloat16 64 40 ()));
      test
        "products from initial values kept apart compute the one kernel's bits"
        (apart (sum_of_products_from Nx.bfloat16 64 40 ()));
    ]

let () =
  exit
    (run "Lower_reduce"
       [
         group "reductions"
           [ extremes; int_folds; float_sums; float_products; complex_sums ];
         group "scans"
           [ extreme_scans; int_scans; float_scans; complex_scans; long_scans ];
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
             blocked_contractions;
             blocked_rows;
             few_rows;
           ];
         sums_of_products;
         work_bounds;
         cases;
       ])
