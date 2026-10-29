(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scalar programs compiled by Rune.jit give the bits nx's eager kernels give
   for every exact kind, any NaN for a NaN, and compile whole: no call is
   evaluated. A narrow float's transcendental kind, computed at float32 and
   rounded once, is within one ulp of the correctly rounded result (rune2.mli,
   Numerics). Lower_real's and Lower_complex's suites hold the bounds at
   float32, float64 and the complex dtypes.

   A case compiles one program over operands of one dtype. Its rows apply one
   kind each to the same operands: the operands are broadcast along a first axis
   of one row per kind, and the program selects row [r]'s kind by the
   coordinate, so that one compilation checks every kind of a family. Each
   operand holds its dtype's special values, every pair of a few of them, then
   random elements. *)

open Windtrap
module D = Nx_array.Dtype
module P = Nx.Prim
module K = Nx_kernel.Prog
module Rune = Rune2
module T = Nx.Ptree

let dtypes =
  D.
    [
      Any Float64;
      Any Float32;
      Any Float16;
      Any Bfloat16;
      Any Float8_e4m3fn;
      Any Float8_e5m2;
      Any Float4_e2m1fn;
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
      Any Complex128;
      Any Complex64;
      Any Bool;
      Any Bit;
    ]

let name (D.Any d) = D.name d
let bits (D.Any d) = D.bits d
let bytes d = max 1 (bits d / 8)
let is_float (D.Any d) = D.is D.Float d
let is_complex (D.Any d) = D.is D.Complex d
let is_bool (D.Any d) = D.is D.Boolean d

let hex b =
  String.concat ""
    (List.init (String.length b) (fun i ->
         Printf.sprintf "%02x" (Char.code b.[String.length b - 1 - i])))

(* Elements

   An element is its bytes in the host's (little-endian) order. *)

let le w (z : int64) =
  String.init w (fun i ->
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical z (8 * i)) 0xffL)))

(* A float format's special codes: zeros, one, the infinity, a quiet NaN, the
   greatest finite and the least subnormal first, which {!operands} pairs, then
   the greatest subnormal, the least normal, one's successor and more NaNs with
   payloads; each of both signs. *)
let format_specials ~e ~m =
  let open Int64 in
  let w = 1 + e + m in
  let bias = sub (shift_left 1L (e - 1)) 1L in
  let exps = shift_left (sub (shift_left 1L e) 1L) m in
  let top = shift_left 1L (m - 1) in
  let mags =
    [
      0L;
      shift_left bias m;
      exps;
      logor exps top;
      sub exps 1L;
      1L;
      sub (shift_left 1L m) 1L;
      shift_left 1L m;
      logor (shift_left bias m) 1L;
      logor exps 1L;
      logor (logor exps top) 1L;
    ]
  in
  let sign = shift_left 1L (w - 1) in
  List.concat_map (fun x -> [ x; logor x sign ]) mags

(* Values at a narrow format's rounding ties and its saturation: 1 between two
   codes, a subnormal tie, the greatest finite and past it. *)
let ties =
  let p = Float.ldexp 1. in
  List.concat_map
    (fun (m, sub, top, past) ->
      [
        1. +. p (-(m + 1));
        1. +. (3. *. p (-(m + 1)));
        p sub;
        1.5 *. p sub;
        top;
        past;
      ])
    [
      (10, -25, 65504., 65520.);
      (7, -134, 0x1.fep127, 0x1.ffp127);
      (3, -10, 448., 464.);
      (2, -17, 57344., 61440.);
      (1, -2, 6., 7.);
      (23, -150, 0x1.fffffep127, 0x1.ffffffp127);
    ]
  @ [ 5.; 2.5; 0.75 ]

(* [x] and the floats one ulp each side of it, as [bits_of] gives them. *)
let around bits_of xs =
  List.concat_map
    (fun x ->
      let b = bits_of x in
      List.concat_map
        (fun b -> [ b; Int64.logor b (bits_of (-0.)) ])
        [ b; Int64.pred b; Int64.succ b ])
    xs

let f32_bits x =
  Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xffffffffL

let int_specials w =
  let open Int64 in
  let min = shift_left 1L (w - 1) and max = sub (shift_left 1L (w - 1)) 1L in
  [
    0L;
    1L;
    -1L;
    min;
    max;
    2L;
    -2L;
    succ min;
    pred max;
    7L;
    -7L;
    add (shift_left 1L 24) 1L;
    add (shift_left 1L 53) 1L;
    add (shift_left 1L 11) 1L;
    add (shift_left 1L 63) 1025L;
    sub (shift_left 1L 32) 1L;
  ]

let codes n = List.init n Int64.of_int

(* A complex dtype's parts beyond its format's specials, each of both signs: the
   unit circle's neighbourhood, where log |z| takes log1p, and magnitudes either
   side of each switch its kinds take: exp's near range, the clamps of e^z and
   2^z, where sinh and cosh are e^|x|/2, where asin turns to its terms at
   infinity, the switch to Payne and Hanek of y and of 2^z's phase y ln2, and
   atanh's overflow guard. *)
let c64_parts =
  [ 0.5; 2.; 0.6; 0.8; 0x1.fffffep-1; 13.; 100.; 200.; 300. ]
  @ [ 12288.; 40000.; 49152.; 0x1p63; 1e30 ]

let c128_parts =
  [ 0.5; 2.; 0.6; 0.8; 0x1.fffffffffffffp-1; 23.; 800.; 1500.; 2200. ]
  @ [ 3e8; 0x1.8p28; 0x1.8p29; 0x1p511; 1e300 ]

(* Leading pairs, which binary kinds also meet as every pair of two: real,
   imaginary and zero numbers, infinite and NaN parts, a negative real. *)
let leading =
  let nan = Float.nan and inf = Float.infinity in
  [
    (1., 0.);
    (0., 1.);
    (-2., 0.);
    (0.5, -0.5);
    (2., -0.);
    (-0., -2.);
    (inf, 0.);
    (0., Float.neg inf);
    (nan, 1.);
    (1., nan);
    (-1.5, 3.);
    (0., 0.);
  ]

(* The leading pairs, then every pair of the format's specials and [parts]. *)
let complex_specials ~e ~m w bits_of parts =
  let signed = List.concat_map (fun x -> [ bits_of x; bits_of (-.x) ]) parts in
  let ps = format_specials ~e ~m @ signed in
  let lead = List.map (fun (re, im) -> le w (bits_of re) ^ le w (bits_of im)) in
  lead leading
  @ List.concat_map (fun re -> List.map (fun im -> le w re ^ le w im) ps) ps

let specials (D.Any dt as d) : string list =
  let w = bytes d in
  let of_list xs = List.map (le w) xs in
  match dt with
  | D.Float64 ->
      of_list (format_specials ~e:11 ~m:52 @ around Int64.bits_of_float ties)
  | D.Float32 -> of_list (format_specials ~e:8 ~m:23 @ around f32_bits ties)
  | D.Float16 -> of_list (format_specials ~e:5 ~m:10)
  | D.Bfloat16 -> of_list (format_specials ~e:8 ~m:7)
  | D.Float8_e4m3fn | D.Float8_e5m2 | D.Uint8 | D.Int8 -> of_list (codes 256)
  | D.Float4_e2m1fn | D.Int4 | D.Uint4 -> of_list (codes 16)
  | D.Bool | D.Bit -> of_list (codes 2)
  | D.Complex64 -> complex_specials ~e:8 ~m:23 4 f32_bits c64_parts
  | D.Complex128 ->
      complex_specials ~e:11 ~m:52 8 Int64.bits_of_float c128_parts
  | _ -> of_list (int_specials (bits d))

(* A random element: random bits, or a float of moderate size. *)
let random (D.Any dt as d) st =
  let w = bytes d in
  let any () = String.init w (fun _ -> Char.chr (Random.State.int st 256)) in
  let moderate () = Random.State.float st 8. -. 4. in
  match dt with
  | D.Bool | D.Bit -> le 1 (Int64.of_int (Random.State.int st 2))
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn ->
      le 1 (Int64.of_int (Random.State.int st 16))
  | _ when Random.State.bool st -> any ()
  | D.Float64 -> K.bits D.Float64 (moderate ())
  | D.Float32 -> K.bits D.Float32 (moderate ())
  | D.Float16 -> K.bits D.Float16 (moderate ())
  | D.Bfloat16 -> K.bits D.Bfloat16 (moderate ())
  | D.Complex64 -> K.bits D.Complex64 { re = moderate (); im = moderate () }
  | D.Complex128 -> K.bits D.Complex128 { re = moderate (); im = moderate () }
  | _ -> le w (Int64.of_int (Random.State.int st 21 - 10))

(* Values *)

type values = Vs : ('v, 's, Nx.host) Nx.t list -> values

let truths es = Array.of_list (List.map (fun e -> e.[0] <> '\000') es)

(* The value of dtype [dt] whose elements' bits are [es]. *)
let value (type v s) (dt : (v, s) D.t) es : (v, s, Nx.host) Nx.t =
  let n = List.length es in
  let raw = String.concat "" es in
  let bytes = Array.init (String.length raw) (fun i -> Char.code raw.[i]) in
  match dt with
  | D.Bool -> Nx.create dt [| n |] (truths es)
  | D.Bit -> Nx.create dt [| n |] (truths es)
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn ->
      Nx.bitcast dt (Nx.create D.Uint4 [| n |] bytes)
  | _ ->
      let w = String.length raw / max 1 n in
      let u = Nx.create D.Uint8 [| n * w |] bytes in
      if w = 1 then Nx.bitcast dt u else Nx.bitcast dt (Nx.reshape [| n; w |] u)

let values (D.Any dt) ess = Vs (List.map (value dt) ess)

(* [counted f] is [f ()], whose compiled calls must compile whole: a declined
   lowering, evaluated by nx, fails. [compiled f x] and [compiled2 f x y] are
   the call of [f] compiled by [Rune.jit]. *)
let counted f =
  let before = Rune.evaluations () in
  let y = f () in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  y

let compiled f x =
  counted (fun () -> Rune.jit T.(tensor @-> returns tensor) f x)

let compiled2 f x y =
  counted (fun () -> Rune.jit T.(tensor @-> tensor @-> returns tensor) f x y)

(* The codes of an 8- or 4-bit dtype of [n] codes that {!operands} pairs: zeros
   of both signs, the extremes and their neighbours; 1, 1.5 and 3 of fp4; 1 of
   both fp8 formats, e5m2's greatest finite and infinity. *)
let paired n =
  let h = n / 2 in
  let more = if n = 16 then [ 3; 5 ] else [ 0x38; 0x3c; 0x7b; 0x7c ] in
  [ 0; 1; 2; h - 2; h - 1; h; h + 1; h + 2; n - 2; n - 1 ] @ more

(* Operands [x] and [y] of [d]: [x] holds the specials, then [x] and [y] hold
   every pair of the first twelve, then random elements. *)
let operands d seed =
  let st = Random.State.make [| seed |] in
  let sp = specials d in
  let few =
    let (D.Any dt) = d in
    match dt with
    | D.Float8_e4m3fn | D.Float8_e5m2 | D.Uint8 | D.Int8 ->
        List.map (List.nth sp) (paired 256)
    | D.Float4_e2m1fn | D.Int4 | D.Uint4 -> List.map (List.nth sp) (paired 16)
    | _ -> List.filteri (fun i _ -> i < 12) sp
  in
  let rand n = List.init n (fun _ -> random d st) in
  let x =
    sp @ List.concat_map (fun a -> List.map (fun _ -> a) few) few @ rand 64
  in
  let y =
    rand (List.length sp) @ List.concat_map (fun _ -> few) few @ rand 64
  in
  (x, y)

(* Programs

   A program over operands [ins] whose output, at row [r] of the first axis, is
   the value [rows.(r)] computes. *)

let select ~ins rows =
  let nodes = ref [] and n = ref 0 in
  let add nd =
    nodes := nd :: !nodes;
    incr n;
    !n - 1
  in
  Array.iteri (fun i _ -> ignore (add (K.In i))) ins;
  let outs = List.map (fun row -> row add) rows in
  let coord = add (K.Coord 0) in
  let last = List.length outs - 1 in
  let pick =
    List.fold_left
      (fun (acc, r) o ->
        let at =
          add (K.Const (D.Any D.Int64, K.bits D.Int64 (Int64.of_int r)))
        in
        let c = add (K.Op2 (Compare Equal, coord, at)) in
        (add (K.Op3 (Where, c, o, acc)), r - 1))
      (List.nth outs last, last - 1)
      (List.tl (List.rev outs))
    |> fst
  in
  K.v ~ins:(Iarray.of_array ins)
    (Iarray.of_list (List.rev !nodes))
    ~outs:(Iarray.of_list [ pick ])

(* The node [i], of [d], as its bits in a uint64, or as itself where no kind
   reads them: a complex128. *)
let raw64 add (D.Any dt as d) i =
  let cast t i = add (K.Op1 (Cast, D.Any t, i)) in
  let bitcast t i = add (K.Op1 (Bitcast, D.Any t, i)) in
  match dt with
  | D.Bool | D.Bit -> cast D.Uint64 i
  | D.Int4 | D.Uint4 | D.Float4_e2m1fn -> cast D.Uint64 (bitcast D.Uint4 i)
  | D.Complex128 -> bitcast D.Uint64 (cast D.Float64 i)
  | _ -> (
      match bits d with
      | 64 -> bitcast D.Uint64 i
      | 32 -> cast D.Uint64 (bitcast D.Uint32 i)
      | 16 -> cast D.Uint64 (bitcast D.Uint16 i)
      | _ -> cast D.Uint64 (bitcast D.Uint8 i))

(* The dtype a result of [d] is compared at: its codes for a narrow float, which
   its decoded value would quiet. *)
let compared (D.Any dt as d) =
  match dt with
  | D.Float16 | D.Bfloat16 -> D.Any D.Uint16
  | D.Float8_e4m3fn | D.Float8_e5m2 -> D.Any D.Uint8
  | D.Float4_e2m1fn -> D.Any D.Uint4
  | _ -> d

let as_compared add d i =
  let c = compared d in
  if name c = name d then i else add (K.Op1 (Bitcast, c, i))

(* Running *)

type row = { label : string; node : (K.node -> int) -> int }

(* Formats: a float dtype's width in bits and whether a code is a NaN. *)

let ieee ~e ~m c =
  let open Int64 in
  let exps = sub (shift_left 1L e) 1L in
  logand (shift_right_logical c m) exps = exps
  && logand c (sub (shift_left 1L m) 1L) <> 0L

let format (D.Any dt) =
  match dt with
  | D.Float64 | D.Complex128 -> Some (64, ieee ~e:11 ~m:52)
  | D.Float32 | D.Complex64 -> Some (32, ieee ~e:8 ~m:23)
  | D.Float16 -> Some (16, ieee ~e:5 ~m:10)
  | D.Bfloat16 -> Some (16, ieee ~e:8 ~m:7)
  | D.Float8_e5m2 -> Some (8, ieee ~e:5 ~m:2)
  | D.Float8_e4m3fn -> Some (8, fun c -> Int64.logand c 0x7FL = 0x7FL)
  | D.Float4_e2m1fn -> Some (4, fun _ -> false)
  | _ -> None

let code b =
  let n = ref 0L in
  for i = String.length b - 1 downto 0 do
    n := Int64.logor (Int64.shift_left !n 8) (Int64.of_int (Char.code b.[i]))
  done;
  !n

(* The distance in ordered ranks of two codes of [w] bits, -0 and +0 one
   apart. *)
let ranks w a b =
  let mask = Int64.pred (Int64.shift_left 1L (w - 1)) in
  let neg c = Int64.logand (Int64.shift_right_logical c (w - 1)) 1L = 1L in
  let ma = Int64.logand a mask and mb = Int64.logand b mask in
  if neg a = neg b then Int64.abs (Int64.sub ma mb)
  else Int64.add (Int64.add ma mb) 1L

(* Whether the elements [a] and [b] of [d] agree within [tol] ranks per part:
   two NaN parts agree whatever their bits. *)
let close d tol a b =
  match format d with
  | None -> a = b
  | Some (w, is_nan) ->
      let parts s =
        if is_complex d then
          let h = String.length s / 2 in
          [ String.sub s 0 h; String.sub s h h ]
        else [ s ]
      in
      let width c =
        if w < 64 then Int64.logand c (Int64.pred (Int64.shift_left 1L w))
        else c
      in
      List.for_all2
        (fun a b ->
          let a = width (code a) and b = width (code b) in
          if is_nan a || is_nan b then is_nan a && is_nan b
          else ranks w a b <= Int64.of_int tol)
        (parts a) (parts b)

(* The rows whose elements differ by more than [tol] ulps, at most five: the
   row's label, the element's operands [ops] and both results. Elements of row
   [r] are compared at the format of [at r]; [skip r j] excuses element [j] of
   row [r], and [exact r j] holds it to 0 ulps. *)
let differences ?(tol = 0) ?(skip = fun _ _ -> false)
    ?(exact = fun _ _ -> false) ~at labels ops ~want ~got =
  let n = match ops with es :: _ -> List.length es | [] -> 1 in
  let bad = ref [] in
  Array.iteri
    (fun i w ->
      let row = i / n and j = i mod n in
      if
        let tol = if exact row j then 0 else tol in
        (not (skip row j || close (at row) tol w got.(i)))
        && List.length !bad < 5
      then
        let operand es = hex (List.nth es j) in
        bad :=
          Printf.sprintf "%s of %s: expected %s, compiled %s"
            (List.nth labels row)
            (String.concat " " (List.map operand ops))
            (hex w)
            (hex got.(i))
          :: !bad)
    want;
  List.rev !bad

(* [p]'s map over the operands [xs], each a row broadcast to the shape [s], into
   [r]. *)
let map_rows p s r xs =
  let n = s.(1) in
  let load x = P.Plain (Nx.broadcast_to s (Nx.reshape [| 1; n |] x)) in
  let y, () =
    P.eval ~by:"t"
      (P.Map
         {
           layout = Nx_array.Layout.contiguous (Iarray.of_array s);
           prog = p;
           outs = [ r ];
           loads = Iarray.of_list (List.map load xs);
           at = None;
         })
  in
  y

(* Runs [p] over the operands [Vs xs], whose elements are [ops], at shape [rows;
   n], eagerly and compiled, and is their differences at [d]. *)
let check ?tol ?at d p labels (Vs xs) ops =
  let at = Option.value at ~default:(fun _ -> d) in
  let (D.Any r) = compared d in
  let n = match ops with es :: _ -> List.length es | [] -> 1 in
  let s = [| List.length labels; n |] in
  let f xs = map_rows p s r xs in
  let got =
    match xs with
    | [ x ] -> compiled (fun x -> f [ x ]) x
    | [ x; y ] -> compiled2 (fun x y -> f [ x; y ]) x y
    | _ -> invalid_arg "check: one or two operands"
  in
  equal ~msg:"shape" (array int) s (Nx.shape got);
  let bits y = Array.map (K.bits r) (Nx.to_array y) in
  differences ?tol ~at labels ops ~want:(bits (f xs)) ~got:(bits got)

let no_difference bad = equal (list string) [] bad

(* Kinds *)

let unaries =
  K.
    [
      (Neg, "neg");
      (Recip, "recip");
      (Abs, "abs");
      (Sign, "sign");
      (Sqrt, "sqrt");
      (Exp, "exp");
      (Exp2, "exp2");
      (Log, "log");
      (Log2, "log2");
      (Log1p, "log1p");
      (Expm1, "expm1");
      (Sin, "sin");
      (Cos, "cos");
      (Tan, "tan");
      (Asin, "asin");
      (Acos, "acos");
      (Atan, "atan");
      (Sinh, "sinh");
      (Cosh, "cosh");
      (Tanh, "tanh");
      (Asinh, "asinh");
      (Acosh, "acosh");
      (Atanh, "atanh");
      (Erf, "erf");
      (Floor, "floor");
      (Ceil, "ceil");
      (Round, "round");
      (Trunc, "trunc");
    ]

let binaries =
  K.
    [
      (Add, "add");
      (Sub, "sub");
      (Mul, "mul");
      (Fdiv, "fdiv");
      (Idiv, "idiv");
      (Mod, "mod");
      (Pow, "pow");
      (Atan2, "atan2");
      (Maximum, "maximum");
      (Minimum, "minimum");
      (And, "and");
      (Or, "or");
      (Xor, "xor");
      (Threefry, "threefry");
    ]

let compares =
  K.
    [
      (Equal, "equal");
      (Not_equal, "not_equal");
      (Less, "less");
      (Less_equal, "less_equal");
    ]

let accepts1 k (D.Any x) (D.Any y) = K.accepts1 k x y
let accepts2 k (D.Any x) = K.accepts2 k x
let accepts3 k (D.Any c) (D.Any x) = K.accepts3 k c x

let finish d rows =
  List.map
    (fun (label, f) -> { label; node = (fun add -> as_compared add d (f add)) })
    rows

let run_rows ?at d ~ins rows vs ops =
  let p = select ~ins (List.map (fun r -> r.node) rows) in
  check ?at d p (List.map (fun r -> r.label) rows) vs ops

(* The kinds a float or complex dtype computes exactly. *)
let exact1 d (k : K.unary) =
  match k with
  | Neg -> true
  | Recip | Abs | Sign | Sqrt | Floor | Ceil | Round | Trunc ->
      not (is_complex d)
  | _ -> not (is_float d || is_complex d)

let exact2 d (k : K.binary) =
  match k with
  | Mul | Fdiv | Mod -> not (is_complex d)
  | Pow | Atan2 -> not (is_float d || is_complex d)
  | _ -> true

let unary_rows ?(keep = exact1) d =
  List.filter_map
    (fun (k, label) ->
      if accepts1 (K.Unary k) d d && keep d k then
        Some (label, fun add -> add (K.Op1 (Unary k, d, 0)))
      else None)
    unaries

(* Every kind of two or three operands of [d], each giving [d]: the comparisons
   cast back to [d], Where by [x < y], Fma of [x], [y], [x]. *)
let binary_rows d =
  let bin =
    List.filter_map
      (fun (k, label) ->
        if accepts2 (K.Binary k) d && exact2 d k then
          Some (label, fun add -> add (K.Op2 (Binary k, 0, 1)))
        else None)
      binaries
  in
  let cmp =
    List.map
      (fun (c, label) ->
        (label, fun add -> add (K.Op1 (Cast, d, add (K.Op2 (Compare c, 0, 1))))))
      compares
  in
  let where =
    ( "where",
      fun add -> add (K.Op3 (Where, add (K.Op2 (Compare Less, 0, 1)), 0, 1)) )
  in
  let fma =
    if accepts3 K.Fma d d then
      [ ("fma", fun add -> add (K.Op3 (Fma, 0, 1, 0))) ]
    else []
  in
  bin @ cmp @ (where :: fma)

let by_dtype ds =
  let pp ppf (d, seed) = Format.fprintf ppf "(%s, %d)" (name d) seed in
  Gen.with_pp pp (Gen.pair (Gen.of_list ds) (Gen.int_range 0 1_000_000))

let classes =
  [
    ("float", is_float);
    ("complex", is_complex);
    ("boolean", is_bool);
    ("integer", fun d -> not (is_float d || is_complex d || is_bool d));
  ]

(* Covers [d]'s name and class, and each class of [ds]. *)
let cover_classes ds d =
  cover (name d) true;
  List.iter
    (fun (label, is) -> if List.exists is ds then cover label (is d))
    classes

let family title ds rows ~arity =
  prop title ~count:2
    ~examples:(List.map (fun d -> (d, 7)) ds)
    (by_dtype ds)
    (fun (d, seed) ->
      cover_classes ds d;
      let x, y = operands d seed in
      let ops = if arity = 1 then [ x ] else [ x; y ] in
      let ins = Array.make arity d in
      no_difference (run_rows d ~ins (finish d (rows d)) (values d ops) ops))

let unary =
  family "exact unary kinds give eager's bits"
    (List.filter (fun d -> unary_rows d <> []) dtypes)
    unary_rows ~arity:1

let binary =
  family "exact binary kinds, comparisons, where and fma give eager's bits"
    dtypes binary_rows ~arity:2

(* A narrow float's transcendental kind computes at float32 and rounds once, so
   it is within one ulp of the correctly rounded result: float64's function of
   the operands, rounded once by nx's cast, and exact where C99 Annex F gives a
   zero, an infinity, ±1 or a NaN. A pow whose result rounds to an infinity, of
   operands not all ±0, ±inf or NaN, is not bound; the formats without an
   infinity saturate, so every pow there is. *)
let narrow = List.filter (fun d -> is_float d && bits d < 32) dtypes

let reference1 : K.unary -> float -> float = function
  | Exp -> Float.exp
  | Exp2 -> Float.exp2
  | Log -> Float.log
  | Log2 -> Float.log2
  | Log1p -> Float.log1p
  | Expm1 -> Float.expm1
  | Sin -> Float.sin
  | Cos -> Float.cos
  | Tan -> Float.tan
  | Asin -> Float.asin
  | Acos -> Float.acos
  | Atan -> Float.atan
  | Sinh -> Float.sinh
  | Cosh -> Float.cosh
  | Tanh -> Float.tanh
  | Asinh -> Float.asinh
  | Acosh -> Float.acosh
  | Atanh -> Float.atanh
  | Erf -> Float.erf
  | _ -> invalid_arg "reference1: an exact kind"

let reference2 : K.binary -> (float -> float -> float) option = function
  | Pow -> Some Float.pow
  | Atan2 -> Some Float.atan2
  | _ -> None

(* [zs] rounded once into [dt], as floats. *)
let round_trip (type v s) (dt : (v, s) D.t) zs =
  let z = Nx.create D.Float64 [| Array.length zs |] zs in
  Nx.to_array (Nx.cast D.Float64 (Nx.cast dt z))

let special v = v = 0. || not (Float.is_finite v)

let floats (type v s) (dt : (v, s) D.t) es =
  Nx.to_array (Nx.cast D.Float64 (value dt es))

(* The codes, at the dtype [compared d] gives, of the floats [zs] rounded once
   into [d]. *)
let rounded_codes (D.Any dt as d) zs =
  let (D.Any c) = compared d in
  let z = Nx.create D.Float64 [| Array.length zs |] zs in
  Array.map (K.bits c) (Nx.to_array (Nx.bitcast c (Nx.cast dt z)))

let rounded =
  let rows d =
    List.filter_map
      (fun (k, label) ->
        if accepts1 (K.Unary k) d d && not (exact1 d k) then
          Some
            ( label,
              (fun add -> add (K.Op1 (Unary k, d, 0))),
              fun x _ -> reference1 k x )
        else None)
      unaries
    @ List.filter_map
        (fun (k, label) ->
          Option.map
            (fun f -> (label, (fun add -> add (K.Op2 (Binary k, 0, 1))), f))
            (reference2 k))
        binaries
  in
  prop
    "narrow floats' transcendental kinds are within one ulp of the correctly \
     rounded result"
    ~count:2
    ~examples:(List.map (fun d -> (d, 7)) narrow)
    (by_dtype narrow)
    (fun ((D.Any dt as d), seed) ->
      cover_classes narrow d;
      let x, y = operands d seed in
      let rows = rows d in
      let labels = List.map (fun (l, _, _) -> l) rows in
      let arity row =
        if List.nth labels row = "pow" || List.nth labels row = "atan2" then 2
        else 1
      in
      let p =
        select ~ins:[| d; d |]
          (List.map (fun (_, node, _) add -> as_compared add d (node add)) rows)
      in
      let (D.Any r) = compared d in
      let n = List.length x in
      let f x y = map_rows p [| List.length rows; n |] r [ x; y ] in
      let got = compiled2 f (value dt x) (value dt y) in
      let xs = floats dt x and ys = floats dt y in
      let float64s =
        Array.concat (List.map (fun (_, _, g) -> Array.map2 g xs ys) rows)
      in
      let rounded = round_trip dt float64s in
      let pow row = List.nth labels row = "pow" in
      let specials row j =
        special xs.(j) && (arity row = 1 || special ys.(j))
      in
      let overflows row j =
        pow row
        && Float.abs rounded.((row * n) + j) = Float.infinity
        && not (specials row j)
      in
      (* Annex F's zeros, infinities, ±1 and NaNs, and pow's 1s. *)
      let exact row j =
        let r = float64s.((row * n) + j) and x = xs.(j) and y = ys.(j) in
        (specials row j && (special r || Float.abs r = 1.))
        || pow row
           && (y = 0. || x = 1. || (x = -1. && Float.abs y = Float.infinity))
      in
      no_difference
        (differences ~tol:1 ~skip:overflows ~exact
           ~at:(fun _ -> d)
           labels [ x; y ] ~want:(rounded_codes d float64s)
           ~got:(Array.map (K.bits r) (Nx.to_array got))))

(* Casts and bitcasts: one row per destination, compared by its bits in a
   uint64, a NaN that a cast into another format makes as any NaN. A complex128
   destination, which no uint64 holds, is compared by its real part there and
   whole by {!into_complex128}. *)

(* The format of a float's or complex's parts, which a cast keeping it keeps bit
   for bit (the real part, from or into a complex). *)
let part (D.Any dt as d) =
  match dt with
  | D.Complex128 -> "float64"
  | D.Complex64 -> "float32"
  | _ when is_float d -> name d
  | _ -> "none: " ^ name d

let conversion title k ~nan ds dsts =
  prop title ~count:2
    ~examples:(List.map (fun d -> (d, 11)) ds)
    (by_dtype ds)
    (fun (d, seed) ->
      cover_classes ds d;
      let x, _ = operands d seed in
      let dsts = dsts d in
      let rows =
        List.map
          (fun t ->
            {
              label = name t;
              node = (fun add -> raw64 add t (add (K.Op1 (k, t, 0))));
            })
          dsts
      in
      let at row =
        let t = List.nth dsts row in
        if (not nan) || part d = part t then D.Any D.Uint64
        else match t with D.Any D.Complex128 -> D.Any D.Float64 | t -> t
      in
      no_difference
        (run_rows ~at (D.Any D.Uint64) ~ins:[| d |] rows (values d [ x ]) [ x ]))

(* A cast into complex128, both parts, a NaN part from another format as any
   NaN. *)
let into_complex128 =
  prop "casts into complex128 give eager's values in both parts" ~count:2
    ~examples:(List.map (fun d -> (d, 13)) dtypes)
    (by_dtype dtypes)
    (fun ((D.Any dt as d), seed) ->
      cover_classes dtypes d;
      let x, _ = operands d seed in
      let into x =
        fst
          (P.eval ~by:"t"
             (P.Map
                {
                  layout =
                    Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape x));
                  prog =
                    K.of_node ~ins:(Iarray.of_list [ d ])
                      (Op1 (Cast, D.Any D.Complex128, 0));
                  outs = [ D.Complex128 ];
                  loads = Iarray.of_list [ P.Plain x ];
                  at = None;
                }))
      in
      let bits y = Array.map (K.bits D.Complex128) (Nx.to_array y) in
      let v = value dt x in
      no_difference
        (differences
           ~at:(fun _ ->
             if part d = "float64" then D.Any D.Uint64 else D.Any D.Complex128)
           [ "complex128" ] [ x ]
           ~want:(bits (into v))
           ~got:(bits (compiled into v))))

let cast =
  conversion "casts give eager's values" K.Cast ~nan:true dtypes (fun _ ->
      dtypes)

let bitcast =
  conversion "bitcasts give eager's bits" K.Bitcast ~nan:false
    (List.filter (fun d -> name d <> "complex128") dtypes)
    (fun d -> List.filter (fun t -> accepts1 K.Bitcast d t) dtypes)

(* Shapes and views *)

let three () = T.(tensor @-> tensor @-> tensor @-> returns (pair tensor tensor))

(* [p]'s two outputs, of [a] and [b], at [shape] over [x], [y] and [z], read in
   place, give eager's bits. *)
let map2 (type a s b r) p (a : (a, s) D.t) (b : (b, r) D.t) shape x y z =
  let map x y z =
    let u, (v, ()) =
      P.eval ~by:"t"
        (P.Map
           {
             layout = Nx_array.Layout.contiguous (Iarray.of_array shape);
             prog = p;
             outs = [ a; b ];
             loads = Iarray.of_list [ P.Plain x; P.Plain y; P.Plain z ];
             at = None;
           })
    in
    (u, v)
  in
  let u, v = map x y z in
  let u', v' = counted (fun () -> Rune.jit (three ()) map x y z) in
  let bits dt xs = Array.to_list (Array.map (fun v -> hex (K.bits dt v)) xs) in
  equal ~msg:"shapes"
    (list (array int))
    [ shape; shape ]
    [ Nx.shape u'; Nx.shape v' ];
  equal ~msg:"first" (list string)
    (bits a (Nx.to_array u))
    (bits a (Nx.to_array u'));
  equal ~msg:"second" (list string)
    (bits b (Nx.to_array v))
    (bits b (Nx.to_array v'))

(* x * y + z into int32, and x - z into float16, over three int32 operands. *)
let two_outputs =
  let i32 = D.Any D.Int32 in
  let open K.E in
  let x = input 0 D.Int32 and y = input 1 D.Int32 and z = input 2 D.Int32 in
  K.of_exprs
    ~ins:(Iarray.of_list [ i32; i32; i32 ])
    (Iarray.of_list [ any (fma x y z); any (cast D.Float16 (x - z)) ])

let ints s xs = Nx.create D.Int32 s (Array.map Int32.of_int xs)
let i32_f16 = map2 two_outputs D.Int32 D.Float16

(* A boolean is its byte: true where it is not zero. Its kinds compute on the
   byte, so a byte read as a boolean keeps it. *)
let bool_bytes () =
  let bytes =
    K.v
      ~ins:(Iarray.of_list [ D.Any D.Bool ])
      (Iarray.of_list [ K.In 0; Op1 (Bitcast, D.Any D.Uint8, 0) ])
      ~outs:(Iarray.of_list [ 1 ])
  in
  let f x =
    fst
      (P.eval ~by:"t"
         (P.Map
            {
              layout = Nx_array.Layout.contiguous [| 4 |];
              prog = bytes;
              outs = [ D.Uint8 ];
              loads = Iarray.of_list [ P.Plain (Nx.bitcast D.Bool x) ];
              at = None;
            }))
  in
  let x = Nx.create D.Uint8 [| 4 |] [| 0; 1; 2; 255 |] in
  equal (array int) [| 0; 1; 2; 255 |] (Nx.to_array (compiled f x))

let shapes =
  group "shapes and views"
    [
      test "a byte read as a boolean keeps its byte" bool_bytes;
      test "a zero-size value" (fun () ->
          let e = ints [| 0 |] [||] in
          i32_f16 [| 0 |] e e e);
      test "a one-element value" (fun () ->
          i32_f16 [| 1 |] (ints [| 1 |] [| -7 |]) (ints [| 1 |] [| 3 |])
            (ints [| 1 |] [| 2049 |]));
      test "transposed, flipped and broadcast operands" (fun () ->
          let x = Nx.transpose (ints [| 3; 2 |] [| 0; 1; 2; 3; 4; 5 |]) in
          let y =
            Nx.flip (ints [| 2; 3 |] [| 2147483647; -2; 3; -4; 5; -6 |])
          in
          let z =
            Nx.broadcast_to [| 2; 3 |] (ints [| 1; 3 |] [| 65519; -2049; 7 |])
          in
          i32_f16 [| 2; 3 |] x y z);
    ]

let () =
  exit
    (run "Lower_arith"
       [ unary; binary; rounded; cast; into_complex128; bitcast; shapes ])
