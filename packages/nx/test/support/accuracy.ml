(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The accuracy nx.mli states for its transcendental functions, on a device:
   each is within its bound of the exact result rounded to the dtype, at float32
   against the host's float64 rounded once, at float64 against the C library's
   long double rounded once where long double is wider, and at the 16- and 8-bit
   floats over every value; the special values of C99's Annex F, NaN for a NaN
   operand, and the sign of a zero are exact at every float dtype. The operands
   are placed on the device, and the oracles computed on the host. *)

open Windtrap

(* Where a function computes: its operands placed there. *)
type device = { put : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let host = { put = Fun.id }
let back x = Nx.place Nx.Placement.host x

type unary = { f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }
type binary = { g : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
type fn = Unary of unary | Binary of binary

(* A function, its bounds in ulps at float32 and float64, as nx.mli states them,
   and its index in the oracle's functions (accuracy_stubs.c). *)
type kind = { name : string; fn : fn; f32 : int; f64 : int; index : int }

let narrow_bound = 1
let unary name f f32 f64 index = { name; fn = Unary f; f32; f64; index }

let kinds =
  [
    unary "exp" { f = Nx.exp } 2 2 0;
    unary "log" { f = Nx.log } 1 2 1;
    unary "log1p" { f = Nx.log1p } 1 2 2;
    unary "expm1" { f = Nx.expm1 } 1 2 3;
    unary "sin" { f = Nx.sin } 2 2 4;
    unary "cos" { f = Nx.cos } 2 2 5;
    unary "tan" { f = Nx.tan } 4 2 6;
    unary "asin" { f = Nx.asin } 2 2 7;
    unary "acos" { f = Nx.acos } 2 2 8;
    unary "atan" { f = Nx.atan } 2 2 9;
    unary "sinh" { f = Nx.sinh } 3 2 10;
    unary "cosh" { f = Nx.cosh } 2 2 11;
    unary "tanh" { f = Nx.tanh } 2 2 12;
    unary "erf" { f = Nx.erf } 2 2 13;
    { name = "pow"; fn = Binary { g = Nx.pow }; f32 = 4; f64 = 2; index = 0 };
    {
      name = "atan2";
      fn = Binary { g = Nx.atan2 };
      f32 = 3;
      f64 = 2;
      index = 1;
    };
  ]

external ldbl_wide : unit -> bool = "nx_test_ldbl_wide"

external ldbl_unary : int -> float array -> float array -> unit
  = "nx_test_ldbl_unary"

external ldbl_binary : int -> float array -> float array -> float array -> unit
  = "nx_test_ldbl_binary"

(* Distance *)

(* The position of the [w]-bit float whose bits are [b] on the line of its
   format's values in increasing order: [+0] at 0, [-0] at -1. *)
let position w b =
  let sign = Int64.shift_left 1L (w - 1) in
  if Int64.logand b sign = 0L then b
  else Int64.(sub (neg (logand b (sub sign 1L))) 1L)

(* The bits of each element of [x], and their width. *)
let bits (type b) (x : (float, b) Nx.t) : int * int64 array =
  match Nx.dtype x with
  | Float64 -> (64, Nx.to_array (Nx.bitcast Nx.int64 x))
  | Float32 ->
      (32, Array.map Int64.of_int32 (Nx.to_array (Nx.bitcast Nx.int32 x)))
  | Float16 | BFloat16 ->
      ( 16,
        Array.map
          (fun v -> Int64.of_int (v land 0xffff))
          (Nx.to_array (Nx.bitcast Nx.int16 x)) )
  | Float8_e4m3 | Float8_e5m2 ->
      (8, Array.map Int64.of_int (Nx.to_array (Nx.bitcast Nx.uint8 x)))

(* The ulps between the elements of [a] and [b] at each index, as positions
   apart, saturated to [max_int]. *)
let distances a b =
  let w, a = bits a and _, b = bits b in
  Array.map2
    (fun a b ->
      let d = Int64.abs (Int64.sub (position w a) (position w b)) in
      if d < 0L || d > Int64.of_int max_int then max_int else Int64.to_int d)
    a b

(* Inputs *)

(* A [w]-bit float tensor of the bit patterns [bs]. *)
let of_bits (type b) (dt : (float, b) Nx.dtype) bs : (float, b) Nx.t =
  let n = [| Array.length bs |] in
  match dt with
  | Float64 -> Nx.bitcast dt (Nx.create Nx.int64 n bs)
  | Float32 ->
      Nx.bitcast dt (Nx.create Nx.int32 n (Array.map Int64.to_int32 bs))
  | Float16 | BFloat16 ->
      Nx.bitcast dt
        (Nx.create Nx.int16 n
           (Array.map (fun b -> (Int64.to_int b lxor 0x8000) - 0x8000) bs))
  | Float8_e4m3 | Float8_e5m2 ->
      Nx.bitcast dt (Nx.create Nx.uint8 n (Array.map Int64.to_int bs))

(* Every [w]-bit pattern, [w] at most 16. *)
let every w = Array.init (1 lsl w) Int64.of_int

(* A draw of a [w]-bit pattern: uniform over every value, NaN and the infinities
   included, or of magnitude between [2^-24] and [2^8], where the functions do
   not saturate. *)
let element w =
  let uniform = if w = 64 then Gen.int64 else Gen.map Int64.of_int32 Gen.int32
  and moderate =
    Gen.map
      (fun x ->
        if w = 64 then Int64.bits_of_float x
        else Int64.of_int32 (Int32.bits_of_float x))
      (Gen.map
         (fun (m, e, neg) ->
           let x = Float.ldexp m e in
           if neg then -.x else x)
         (Gen.triple (Gen.float_range 0.5 1.) (Gen.int_range (-23) 8) Gen.bool))
  in
  Gen.frequency [ (1, uniform); (1, moderate) ]

let drawn gen = Gen.array ~size:(Gen.int_range 1 256) gen

(* The check *)

(* Whether each element of [x] is a signaling NaN: its exponent all ones and its
   significand's top bit clear, but not all of its significand. The float8
   formats have none. *)
let signaling (type b) (x : (float, b) Nx.t) =
  let exp, quiet =
    match Nx.dtype x with
    | Float64 -> (0x7ff0_0000_0000_0000L, 0x0008_0000_0000_0000L)
    | Float32 -> (0x7f80_0000L, 0x0040_0000L)
    | BFloat16 -> (0x7f80L, 0x0040L)
    | Float16 -> (0x7c00L, 0x0200L)
    | Float8_e4m3 | Float8_e5m2 -> (0L, 0L)
  in
  let mant = Int64.(sub quiet 1L) in
  Array.map
    (fun b ->
      exp <> 0L
      && Int64.logand b exp = exp
      && Int64.logand b quiet = 0L
      && Int64.logand b mant <> 0L)
    (snd (bits x))

(* The bound's verdict on [got] against [oracle] at each index of [inputs]: NaN
   exactly where the oracle is, and otherwise at most [bound] ulps from it.
   Where an operand is a signaling NaN, the result is NaN or the oracle's, as
   nx.mli allows. The message names the worst input. *)
let within ~bound ~inputs got oracle =
  let g = Nx.to_array got and o = Nx.to_array oracle in
  let d = distances got oracle in
  let snan =
    List.fold_left
      (fun acc x -> Array.map2 ( || ) acc (signaling x))
      (Array.make (Array.length g) false)
      inputs
  in
  let pp_inputs i =
    String.concat ", "
      (List.map (fun x -> Printf.sprintf "%h" (Nx.to_array x).(i)) inputs)
  in
  let worst = ref 0 and at = ref (-1) in
  Array.iteri
    (fun i o ->
      let msg what =
        Printf.sprintf "%s at (%s): got %h, oracle %h" what (pp_inputs i) g.(i)
          o
      in
      if snan.(i) then
        equal ~msg:(msg "signaling NaN") bool true
          (Float.is_nan g.(i) || d.(i) = 0)
      else if Float.is_nan o <> Float.is_nan g.(i) then
        equal ~msg:(msg "NaN") bool (Float.is_nan o) (Float.is_nan g.(i))
      else if (not (Float.is_nan o)) && d.(i) > !worst then begin
        worst := d.(i);
        at := i
      end)
    o;
  if !at >= 0 then
    at_most
      ~msg:
        (Printf.sprintf "ulps at (%s): got %h, oracle %h" (pp_inputs !at)
           g.(!at) o.(!at))
      int ~than:bound !worst

(* The exact result [exact], at float64, rounded to [dt]. float8_e4m3, which has
   no infinity, is NaN where the result rounded to float32, at which it
   computes, is infinite, as nx.mli states. *)
let rounded (type b) (dt : (float, b) Nx.dtype) exact : (float, b) Nx.t =
  match dt with
  | Float8_e4m3 ->
      Nx.where
        (Nx.isinf (Nx.cast Nx.float32 exact))
        (Nx.full_like (Nx.cast dt exact) Float.nan)
        (Nx.cast dt exact)
  | _ -> Nx.cast dt exact

(* [k] at the dtype of [xs] (and [ys]) against its oracle: the host's float64
   rounded once, or the long double one at float64. *)
let check (type b) ~on k ~bound (xs : (float, b) Nx.t) ys =
  let dt = Nx.dtype xs in
  let wide x = Nx.cast Nx.float64 x in
  let (got, oracle) : (float, b) Nx.t * (float, b) Nx.t =
    match (k.fn, ys, dt) with
    | Unary { f }, _, Float64 ->
        let a = Nx.to_array xs in
        let o = Array.make (Array.length a) 0. in
        ldbl_unary k.index a o;
        (back (f (on.put xs)), Nx.create Nx.float64 (Nx.shape xs) o)
    | Binary { g }, Some ys, Float64 ->
        let a = Nx.to_array xs and b = Nx.to_array ys in
        let o = Array.make (Array.length a) 0. in
        ldbl_binary k.index a b o;
        (back (g (on.put xs) (on.put ys)), Nx.create Nx.float64 (Nx.shape xs) o)
    | Unary { f }, _, _ -> (back (f (on.put xs)), rounded dt (f (wide xs)))
    | Binary { g }, Some ys, _ ->
        (back (g (on.put xs) (on.put ys)), rounded dt (g (wide xs) (wide ys)))
    | Binary _, None, _ -> invalid_arg "a binary function of one operand"
  in
  within ~bound ~inputs:(xs :: Option.to_list ys) got oracle

(* Bounds *)

(* Operands of [w] bits, drawn in pairs, printed as their bits. *)
let operands w =
  Gen.with_pp
    (fun ppf ps ->
      Array.iter (fun (a, b) -> Format.fprintf ppf "(0x%Lx, 0x%Lx) " a b) ps)
    (drawn (Gen.pair (element w) (element w)))

(* [k] at the [w]-bit float [dt] on the first operands, or both, of [ps]. *)
let on_operands ~on k dt ~bound ps =
  let xs = of_bits dt (Array.map fst ps) in
  match k.fn with
  | Unary _ -> check ~on k ~bound xs None
  | Binary _ -> check ~on k ~bound xs (Some (of_bits dt (Array.map snd ps)))

let at_float32 ~on k =
  prop
    (k.name ^ " at float32, against float64 rounded once")
    (operands 32)
    (on_operands ~on k Nx.float32 ~bound:k.f32)

let at_float64 ~on k =
  prop (k.name ^ " at float64, against long double rounded once") (operands 64)
    (fun ps ->
      if not (ldbl_wide ()) then
        skip ~reason:"long double is double on this system, no oracle" ();
      on_operands ~on k Nx.float64 ~bound:k.f64 ps)

type narrow = Narrow : string * (float, 'b) Nx.dtype * int -> narrow

let narrows =
  [
    Narrow ("float16", Nx.float16, 16);
    Narrow ("bfloat16", Nx.bfloat16, 16);
    Narrow ("float8_e4m3", Nx.float8_e4m3, 8);
    Narrow ("float8_e5m2", Nx.float8_e5m2, 8);
  ]

(* Every value of a narrow float, and every pair of a float8 one; pairs of a
   16-bit one are drawn. *)
let at_narrow ~on k (Narrow (name, dt, w)) =
  let title = Printf.sprintf "%s at %s, every value" k.name name in
  match k.fn with
  | Unary _ ->
      test title (fun () ->
          check ~on k ~bound:narrow_bound (of_bits dt (every w)) None)
  | Binary _ when w = 8 ->
      let n = 1 lsl w in
      let a = Array.init (n * n) (fun i -> Int64.of_int (i / n))
      and b = Array.init (n * n) (fun i -> Int64.of_int (i mod n)) in
      test (Printf.sprintf "%s at %s, every pair" k.name name) (fun () ->
          check ~on k ~bound:narrow_bound (of_bits dt a) (Some (of_bits dt b)))
  | Binary _ ->
      let pairs =
        Gen.array ~size:(Gen.int_range 1 1024)
          (Gen.pair (Gen.int_range 0 0xffff) (Gen.int_range 0 0xffff))
      in
      prop (Printf.sprintf "%s at %s, pairs" k.name name) pairs (fun ps ->
          let a = Array.map (fun (a, _) -> Int64.of_int a) ps
          and b = Array.map (fun (_, b) -> Int64.of_int b) ps in
          check ~on k ~bound:narrow_bound (of_bits dt a) (Some (of_bits dt b)))

(* Exact values *)

(* Annex F's special values, NaN for NaN and signed zeros: the operands and the
   result, by function. *)
let specials =
  let inf = infinity and ninf = neg_infinity in
  let odd zeros = List.map (fun z -> ([ z ], z)) [ 0.; -0. ] @ zeros in
  [
    ( "exp",
      [
        ([ ninf ], 0.);
        ([ inf ], inf);
        ([ 0. ], 1.);
        ([ -0. ], 1.);
        ([ nan ], nan);
      ] );
    ( "log",
      [
        ([ 0. ], ninf);
        ([ -0. ], ninf);
        ([ 1. ], 0.);
        ([ -1. ], nan);
        ([ ninf ], nan);
        ([ inf ], inf);
        ([ nan ], nan);
      ] );
    ( "log1p",
      odd
        [
          ([ -1. ], ninf);
          ([ -2. ], nan);
          ([ ninf ], nan);
          ([ inf ], inf);
          ([ nan ], nan);
        ] );
    ("expm1", odd [ ([ ninf ], -1.); ([ inf ], inf); ([ nan ], nan) ]);
    ("sin", odd [ ([ inf ], nan); ([ ninf ], nan); ([ nan ], nan) ]);
    ( "cos",
      [
        ([ 0. ], 1.);
        ([ -0. ], 1.);
        ([ inf ], nan);
        ([ ninf ], nan);
        ([ nan ], nan);
      ] );
    ("tan", odd [ ([ inf ], nan); ([ ninf ], nan); ([ nan ], nan) ]);
    ( "asin",
      odd [ ([ 2. ], nan); ([ -2. ], nan); ([ inf ], nan); ([ nan ], nan) ] );
    ( "acos",
      [
        ([ 1. ], 0.);
        ([ 2. ], nan);
        ([ -2. ], nan);
        ([ ninf ], nan);
        ([ nan ], nan);
      ] );
    ("atan", odd [ ([ nan ], nan) ]);
    ("sinh", odd [ ([ inf ], inf); ([ ninf ], ninf); ([ nan ], nan) ]);
    ( "cosh",
      [
        ([ 0. ], 1.);
        ([ -0. ], 1.);
        ([ inf ], inf);
        ([ ninf ], inf);
        ([ nan ], nan);
      ] );
    ("tanh", odd [ ([ inf ], 1.); ([ ninf ], -1.); ([ nan ], nan) ]);
    ("erf", odd [ ([ inf ], 1.); ([ ninf ], -1.); ([ nan ], nan) ]);
    ( "pow",
      [
        ([ nan; 0. ], 1.);
        ([ 2.; -0. ], 1.);
        ([ ninf; 0. ], 1.);
        ([ 1.; nan ], 1.);
        ([ 1.; inf ], 1.);
        ([ 1.; -3. ], 1.);
        ([ 0.; -3. ], inf);
        ([ -0.; -3. ], ninf);
        ([ 0.; -2. ], inf);
        ([ -0.; -2. ], inf);
        ([ -0.; ninf ], inf);
        ([ 0.; 3. ], 0.);
        ([ -0.; 3. ], -0.);
        ([ -0.; 2. ], 0.);
        ([ -1.; inf ], 1.);
        ([ -1.; ninf ], 1.);
        ([ 0.5; ninf ], inf);
        ([ 2.; ninf ], 0.);
        ([ 0.5; inf ], 0.);
        ([ 2.; inf ], inf);
        ([ ninf; -3. ], -0.);
        ([ ninf; -2. ], 0.);
        ([ ninf; 3. ], ninf);
        ([ ninf; 2. ], inf);
        ([ inf; -1. ], 0.);
        ([ inf; 1. ], inf);
        ([ -2.; 0.5 ], nan);
        ([ nan; 1. ], nan);
      ] );
    ( "atan2",
      [
        ([ 0.; 0. ], 0.);
        ([ -0.; 0. ], -0.);
        ([ 0.; 2. ], 0.);
        ([ -0.; 2. ], -0.);
        ([ 2.; inf ], 0.);
        ([ -2.; inf ], -0.);
        ([ nan; 1. ], nan);
        ([ 1.; nan ], nan);
      ] );
  ]

type dtype = Dtype : string * (float, 'b) Nx.dtype -> dtype

let dtypes =
  Dtype ("float32", Nx.float32)
  :: Dtype ("float64", Nx.float64)
  :: List.map (fun (Narrow (n, dt, _)) -> Dtype (n, dt)) narrows

let has_infinities (type b) (dt : (float, b) Nx.dtype) =
  match dt with Float8_e4m3 -> false | _ -> true

let exact ~on k (Dtype (name, dt)) =
  let rows =
    List.filter
      (fun (xs, r) ->
        has_infinities dt
        || List.for_all Float.is_finite
             (r :: List.filter (Fun.negate Float.is_nan) xs))
      (List.assoc k.name specials)
  in
  let pp (xs, _) = String.concat ", " (List.map (Printf.sprintf "%h") xs) in
  cases
    ~name:(fun c -> Printf.sprintf "%s (%s)" k.name (pp c))
    (Printf.sprintf "%s at %s" k.name name)
    rows
    (fun (xs, r) ->
      let t x = Nx.create dt [| 1 |] [| x |] in
      let got =
        match (k.fn, xs) with
        | Unary { f }, [ x ] -> back (f (on.put (t x)))
        | Binary { g }, [ x; y ] -> back (g (on.put (t x)) (on.put (t y)))
        | _ -> invalid_arg "an operand list of another arity"
      in
      let g = (Nx.to_array got).(0) in
      if Float.is_nan r then
        equal ~msg:(Printf.sprintf "%h is NaN" g) bool true (Float.is_nan g)
      else equal ~msg:"bit for bit" int 0 (distances got (t r)).(0))

(* The groups of tests of the functions computed on [on]. *)
let groups on =
  [
    group "float32" (List.map (at_float32 ~on) kinds);
    group "float64" (List.map (at_float64 ~on) kinds);
    group "narrow floats"
      (List.concat_map (fun k -> List.map (at_narrow ~on k) narrows) kinds);
    group "exact values"
      (List.concat_map (fun k -> List.map (exact ~on k) dtypes) kinds);
  ]
