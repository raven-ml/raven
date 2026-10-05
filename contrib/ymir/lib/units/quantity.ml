(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type 'p t = { unit : Unit.t; payload : 'p }

let walk c q =
  Nx.Ptree.Walk.case c (Unit.to_string q.unit);
  { unit = q.unit; payload = Nx.Ptree.Walk.leaf c q.payload }

let is_boolean (type a b) (x : (a, b) Nx.t) =
  match Nx.dtype x with Bool | Bit -> true | _ -> false

let dtype_name x = Nx_dtype.to_string (Nx.dtype x)

let v unit payload =
  if is_boolean payload then
    invalid_arg
      (strf "Quantity.v: a %s tensor has no unit" (dtype_name payload));
  { unit; payload }

let unit q = q.unit
let times u q = { q with unit = Unit.(q.unit * u) }
let per u q = { q with unit = Unit.(q.unit / u) }

(* Messages about elements *)

(* [element_name d i] names the element at index [i] of a payload of dtype [d],
   or the payload when it is a scalar: ["element [41] of an int32 payload"]. *)
let element_name d i =
  let dt = Nx_dtype.to_string d in
  let article = if String.starts_with ~prefix:"int" dt then "an" else "a" in
  if Array.length i = 0 then strf "%s %s payload" article dt
  else
    let index =
      String.concat "; " (Array.to_list (Array.map string_of_int i))
    in
    strf "element [%s] of %s %s payload" index article dt

(* Conversion *)

(* [scale_components d x f] scales the real and imaginary parts of [x] by [f]'s
   real part, with one multiply of its components read as [d]. A complex product
   by [f + 0i] would multiply an infinite part by the zero imaginary part and
   give NaN. *)
let scale_components (type b c) (d : (float, b) Nx.dtype)
    (x : (Complex.t, c) Nx.t) (f : Complex.t) =
  Nx.bitcast (Nx.dtype x) (Nx.mul_s (Nx.bitcast d x) f.re)

(* [int_range d f] is the range of [d] divided by [f], rounded inwards, and
   [f]'s text, for the dtypes whose elements are OCaml ints. *)
let int_range (type b) (d : (int, b) Nx.dtype) f =
  (Nx_dtype.min_value d / f, Nx_dtype.max_value d / f, string_of_int f)

(* [scale fn ~from ~into x f] is [x] times [f], the conversion from [from] to
   [into] in [x]'s dtype. An integer element overflows iff it is outside the
   dtype's range divided by [f], so every element is checked against that range
   first. *)
let scale (type a b) fn ~from ~into (x : (a, b) Nx.t) (f : a) : (a, b) Nx.t =
  let d = Nx.dtype x in
  let checked ((lo : a), (hi : a), factor) =
    let ok = Nx.logical_and (Nx.greater_equal_s x lo) (Nx.less_equal_s x hi) in
    Nx.check ok (fun i ->
        strf "%s: %s overflows converting %s to %s (factor %s)" fn
          (element_name d i) (Unit.to_string from) (Unit.to_string into) factor);
    Nx.mul_s x f
  in
  match d with
  | Nx_dtype.Complex64 -> scale_components Nx.float32 x f
  | Complex128 -> scale_components Nx.float64 x f
  | Int4 -> checked (int_range d f)
  | UInt4 -> checked (int_range d f)
  | Int8 -> checked (int_range d f)
  | UInt8 -> checked (int_range d f)
  | Int16 -> checked (int_range d f)
  | UInt16 -> checked (int_range d f)
  | Int32 -> checked Int32.(div min_int f, div max_int f, to_string f)
  | UInt32 -> checked (0l, Int32.unsigned_div (-1l) f, strf "%lu" f)
  | Int64 -> checked Int64.(div min_int f, div max_int f, to_string f)
  | UInt64 -> checked (0L, Int64.unsigned_div (-1L) f, strf "%Lu" f)
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
      Nx.mul_s x f
  | Bool | Bit -> assert false (* [Unit.ratio_named] refuses booleans. *)

(* [value_named fn u q] is [value u q], its errors naming [fn]. *)
let value_named fn u q =
  let x = q.payload in
  (* A boolean payload in its own unit goes on to [Unit.ratio_named], which
     raises on it. *)
  if Unit.equal q.unit u && not (is_boolean x) then x
  else
    let f = Unit.ratio_named fn (Nx.dtype x) q.unit u in
    scale fn ~from:q.unit ~into:u x f

let value u q = value_named "Quantity.value" u q
let convert u q = { unit = u; payload = value_named "Quantity.convert" u q }

(* Maps *)

let map f q =
  let payload = f q.payload in
  if is_boolean payload then
    invalid_arg
      (strf "Quantity.map: the function returns a %s tensor"
         (dtype_name payload));
  { q with payload }

let map2_named fn f a b =
  { a with payload = f a.payload (value_named fn a.unit b) }

let map2 f a b = map2_named "Quantity.map2" f a b

(* Algebra *)

let add a b = map2_named "Quantity.add" Nx.add a b
let sub a b = map2_named "Quantity.sub" Nx.sub a b

let mul a b =
  let unit = Unit.(a.unit * b.unit) in
  { unit; payload = Nx.mul a.payload b.payload }

let div a b =
  let unit = Unit.(a.unit / b.unit) in
  { unit; payload = Nx.div a.payload b.payload }

(* [power x n] is [x] to the power [n >= 1] by binary powering: products, which
   wrap for integers, and an exponent that is never rounded to [x]'s dtype. *)
let rec power x n =
  if n = 1 then x
  else
    let half = power (Nx.mul x x) (n / 2) in
    if n land 1 = 0 then half else Nx.mul x half

let pow n q =
  let x = q.payload in
  let d = Nx.dtype x in
  if n < 0 && Nx_dtype.is_int d then
    invalid_arg
      (strf "Quantity.pow: %d is negative and the payload is %s" n
         (Nx_dtype.to_string d));
  let unit = Unit.(q.unit ** n) in
  let payload =
    if n = 0 then Nx.ones_like x
    else if n > 0 then power x n
    else
      (* x^n is 1 / (x x^(-n-1)), and [-n - 1] never overflows. *)
      let k = -(n + 1) in
      Nx.recip (if k = 0 then x else Nx.mul x (power x k))
  in
  { unit; payload }

(* [newton n x r] refines [r], an [n]-th root of [x >= 0] whose exponent [1/n]
   was rounded, by one Newton step: [r - (r - x / r^(n-1)) / n]. The error
   squares, so a root a few ulps off ends within about one. A zero, infinite or
   NaN [r] is kept, and the division then reads 1, so neither branch of the
   [where] has a NaN derivative. *)
let newton n x r =
  let ok = Nx.logical_and (Nx.isfinite r) (Nx.not_equal_s r 0.) in
  let k = Float.of_int n in
  let den = Nx.where ok (Nx.pow_s r (k -. 1.)) (Nx.ones_like r) in
  let step = Nx.div_s (Nx.sub r (Nx.div x den)) k in
  Nx.where ok (Nx.sub r step) r

(* [real_root n x] is the real [n]-th root of [x], by [Nx.pow] at the exponent
   [1/n] rounded to [x]'s dtype, then by one Newton step from [n = 3]. *)
let real_root n x =
  let root x =
    let r = Nx.pow_s x (1. /. Float.of_int n) in
    if n = 2 then r else newton n x r
  in
  if n land 1 = 0 then root x
  else
    (* The root of a negative element is minus the root of its opposite. Both
       branches take a non-negative base, so neither has a NaN derivative. *)
    let negative = Nx.less_s x 0. in
    let r = root (Nx.where negative (Nx.neg x) x) in
    Nx.where negative (Nx.neg r) r

let root (type b) n (q : (float, b) Nx.t t) =
  if n < 1 then invalid_arg (strf "Quantity.root: %d is below 1" n);
  let unit = Unit.root n q.unit in
  let x = q.payload in
  let d = Nx.dtype x in
  (* NaN is not below 0, so it passes and its root is NaN. *)
  if n land 1 = 0 then
    Nx.check
      (Nx.logical_not (Nx.less_s x 0.))
      (fun i ->
        strf "Quantity.root: %s is below 0, whose root of order %d is not real"
          (element_name d i) n);
  (* A dtype narrower than float32 would round the exponent: 1/3 is 0.34375 in
     float8_e4m3, which takes 27 to 3.25. Such a payload is rooted in float32
     and cast back. *)
  let payload =
    match d with
    | Float16 | BFloat16 | Float8_e4m3 | Float8_e5m2 ->
        Nx.cast d (real_root n (Nx.cast Nx.float32 x))
    | Float32 | Float64 -> real_root n x
  in
  { unit; payload }

let pp ppf q =
  Format.fprintf ppf "@[<hov>%a@ %a@]" Nx.pp q.payload Unit.pp q.unit
