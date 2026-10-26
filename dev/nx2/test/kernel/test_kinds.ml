(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx_kinds.h: the transcendental kinds within their bounds, f32 at every
   argument and f64 at the goldens' points; the exact kinds against their
   meaning, with NaN, -0, infinities, subnormals and integer extremes drawn;
   the complex transcendental kinds at the goldens' points, at C99 Annex G's
   special values and by their symmetries. *)

open Windtrap
open Exact
module K = Nx_kinds_support

let strf = Printf.sprintf

(* f32 values travel as their bits, f64 values as floats *)

let bits32 x = Int32.to_int (Int32.bits_of_float x) land 0xFFFF_FFFF
let f b = Int32.float_of_bits (Int32.of_int b)
let is_nan b = b land 0x7FFF_FFFF > 0x7F80_0000
let bits64 = Int64.bits_of_float
let pp32 ppf b = Format.fprintf ppf "%h (0x%08x)" (f b) b

(* Bit for bit. *)
let exact32 = Testable.make ~pp:pp32 ~equal:( = )

(* Bit for bit, but any NaN for a NaN: an operation's own NaN differs between
   instruction sets. *)
let value32 =
  Testable.make ~pp:pp32 ~equal:(fun a b -> a = b || (is_nan a && is_nan b))

let value64 =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h (0x%016Lx)" x (bits64 x))
    ~equal:(fun a b ->
      Int64.equal (bits64 a) (bits64 b) || (Float.is_nan a && Float.is_nan b))


(* f32 transcendentals

   gen/sweep_kinds.exe checks every argument and records each kind's largest
   error and worst arguments, stamped with the digest of nx_kinds.h and
   nx_kinds_real.h; a record older than them fails here, and the worst arguments
   are rechecked. A change elsewhere that moves a result, in nx_dtype.h's bit
   helpers, the flags or the compiler, moves the digests of results below
   instead: then too, run the sweep and accept them.

   A test run checks every 256th bit pattern from an offset the run's seed
   draws: every binade of both signs gets 2^15 points, so every region wider
   than 256 ulps is reached by arithmetic. The strata reach the narrower ones:
   each reduction boundary and clamp with its neighbours. Bounds are the
   header's; each kind's fractional error, against the reference itself in
   binary32 ulps at it, is held to Nx_kinds_support.f32_fractions, so that no
   change worsens a result within its rank. *)

let worst_error (w : K.worst) =
  if Array.length w.errors = 0 then 0 else fst w.errors.(0)

(* Fractions are compared at the record's three decimals. *)
let thousandths x = Float.round (x *. 1000.) /. 1000.

let at_bound kind bound (w : K.worst) =
  let e = worst_error w in
  let p = if e = 0 then 0 else snd w.errors.(0) in
  at_most int ~than:bound e ~msg:(strf "%s at %h (0x%08x)" kind (f p) p);
  at_most float_exact
    ~than:(List.assoc kind K.f32_fractions)
    (thousandths w.frac)
    ~msg:(strf "%s's fractional error at %h (0x%08x)" kind (f w.frac_at) w.frac_at)

let test_record =
  test "the full sweep is the headers'" (fun () ->
      let digest, sweeps = K.read_record () in
      if digest <> K.header_digest () then
        failf
          "nx_kinds.h or nx_kinds_real.h changed since the full sweep: run the \
           full sweep (gen/sweep_kinds.exe, from dev/nx2/test/kernel)";
      equal (list string)
        (List.map fst K.f32_bounds)
        (List.map (fun (k, _, _, _) -> k) sweeps);
      List.iter
        (fun (kind, e, (frac, at), ps) ->
          let bound = List.assoc kind K.f32_bounds in
          at_most int ~than:bound e ~msg:kind;
          at_most float_exact ~than:(List.assoc kind K.f32_fractions) frac
            ~msg:kind;
          at_bound kind bound (K.points kind (Array.of_list (at :: ps)));
          equal float_exact frac (thousandths (K.points kind [| at |]).frac)
            ~msg:(strf "%s at 0x%08x" kind at))
        sweeps)

let test_strided =
  List.map
    (fun (kind, bound) ->
      prop (strf "%s at every 256th argument" kind)
        ~count:1 (Gen.int_range 0 255) (fun offset ->
          at_bound kind bound (K.sweep ~step:256 ~offset kind)))
    K.f32_bounds

let test_strata =
  cases ~name:fst "f32 at the strata" K.f32_bounds (fun (kind, bound) ->
      Array.iter
        (fun (region, n, w) ->
          greater int ~than:0 n ~msg:(strf "%s: %s is empty" kind region);
          at_bound kind bound w)
        (K.strata kind))

(* Every value of each narrow float, computed in f32 and rounded once, within 1
   ulp of the exact value rounded once; pow and atan2 likewise at the pairs
   Nx_kinds_support.narrow_binary takes. *)
let narrow_dtypes =
  [
    ("float16", 2);
    ("bfloat16", 3);
    ("float8_e4m3fn", 4);
    ("float8_e5m2", 5);
    ("float4_e2m1fn", 6);
  ]

let test_narrow =
  cases ~name:fst "narrow floats at every value" K.f32_bounds (fun (kind, _) ->
      List.iter
        (fun (name, code) ->
          let e, c = K.narrow kind code in
          at_most int ~than:1 e ~msg:(strf "%s at %s code 0x%x" kind name c))
        narrow_dtypes)

let test_narrow_binary =
  cases ~name:Fun.id "narrow floats at pairs of values" [ "pow"; "atan2" ]
    (fun kind ->
      List.iter
        (fun (name, code) ->
          let e, a, b = K.narrow_binary kind code in
          at_most int ~than:1 e
            ~msg:(strf "%s at %s codes 0x%x 0x%x" kind name a b))
        narrow_dtypes)

(* f32 pow and atan2 at a million pairs, drawn from the run's seed, against the
   C library's f64 functions rounded. *)
let test_binary_transcendentals =
  List.map
    (fun (kind, bound) ->
      prop (strf "%s f32 at drawn pairs" kind) ~count:1 Gen.int (fun seed ->
          let e, a, b = K.binary kind ~seed 1_000_000 in
          at_most int ~than:bound e
            ~msg:(strf "%s %h %h (0x%08x 0x%08x)" kind (f a) (f b) a b)))
    [ ("pow", 4); ("atan2", 3) ]

(* What every library computing a kind through nx_kinds.h gives, as a digest
   of its results over the strata: the same on each target this CPU runs. *)
let test_digests =
  test "digests of every kind over the strata" (fun () ->
      let table target =
        List.concat_map
          (fun (kind, tys) ->
            List.map
              (fun ty -> strf "%s %s %s" kind ty (K.digest ~target kind ty))
              tys)
          K.kinds
        |> String.concat "\n"
      in
      match K.targets () with
      | [] -> fail "no target"
      | first :: rest ->
          let t = table first in
          List.iter (fun target -> equal text t (table target) ~msg:target) rest;
          expect t
          @@ __POS_OF__
               {|
            exp f32 17d2c3c91d74a5e6
            exp f64 9e4592740004bee2
            exp2 f32 162ffc537d0be92c
            exp2 f64 619776aadfc58ecf
            expm1 f32 ed478fe1f7c48a44
            expm1 f64 0fe1a90f7088a3a9
            log f32 ed88f9f51d1056d4
            log f64 5c7f9141f6659317
            log2 f32 7e7bf05e4e06c906
            log2 f64 46fb31d9f7137087
            log1p f32 7c884cacc3dfac9c
            log1p f64 d5b0989c733c0842
            sin f32 3306fe5ff79f397b
            sin f64 cb09ba85cbe1edb3
            cos f32 e505ac6eeb8cd622
            cos f64 147a0f2b6644a9b8
            tan f32 7bbf9342404139c7
            tan f64 6151db99765d791d
            asin f32 fce14c3b05ab0855
            asin f64 e3905a2f07ceeb69
            acos f32 4b8d8de411e65924
            acos f64 ac901c34a3c7530b
            atan f32 9bcb943b590e776a
            atan f64 045759334a6d727d
            sinh f32 4b4aa3fb457a75c0
            sinh f64 2feb552b86b831b7
            cosh f32 f5ff656dcb45f630
            cosh f64 fb7b96d81e8a7f2a
            tanh f32 9936f3800cdafefa
            tanh f64 7bbebdbbc3b03c9d
            erf f32 c26fe6bfe9fb6c9b
            erf f64 1fd4151bac52806d
            sqrt f32 ee9f8255fb18c10d
            sqrt f64 eada3c7c0715ecb9
            floor f32 7de2b62c3fef8114
            floor f64 8bf1717db53997f2
            ceil f32 7606f4f30c6e2999
            ceil f64 bc13e2a9a2892d6e
            round f32 d30d2b349b43fc05
            round f64 41d8cb307a0ab148
            trunc f32 34f9bef2866a88be
            trunc f64 0ecd64f3ca0f53b8
            fdiv f32 b871ece0f2592852
            fdiv f64 e7de96b328ce3382
            atan2 f32 514e43c0d1518dea
            atan2 f64 44753a3f36c3f4f0
            neg f32 cea83e0bddaa4962
            neg f64 ced5a51f9294a4ca
            neg int 51a191f53249a86d
            abs f32 08dcd90d759eb6e2
            abs f64 a8c23c6593f7a14a
            abs int b38dd2f56f9d9e5a
            sign f32 17c4b561ec24f905
            sign f64 57bb9381be03ba45
            sign int 5dea5e3027706bbc
            recip f32 d51e78b9124791bd
            recip f64 c910225a1f5577d9
            recip int b301af41e863f911
            add f32 c6dba66e8ad54c69
            add f64 77a90934e6682cc5
            add int e43acf1b1b03de31
            sub f32 69550897d5cadd4d
            sub f64 ee98dad1fd8c3140
            sub int 428a370b690eeda1
            mul f32 8c47b9c0f9fbf0b6
            mul f64 85ab9ebe7ff8f131
            mul int f6e0a9633e92af59
            mod f32 977b5a2f16e84e40
            mod f64 e2da65fabc2da284
            mod int 8f21a2076f3b02f1
            pow f32 0afbc574906c7d57
            pow f64 7571971ed918fdcf
            pow int c21a5c07dfddd9d7
            maximum f32 a1447ba947915d66
            maximum f64 7bd64cad6c3f7d68
            maximum int 31e15067e696aeb0
            minimum f32 967e301ad5ff4ca8
            minimum f64 23ff51bf4210736c
            minimum int 958d0ad38bd04ad4
            equal f32 803ad2f4668a04f8
            equal f64 940226b225d8cfd8
            equal int 5d0597fa23cf4465
            not_equal f32 05567728a31db838
            not_equal f64 a9e59c8d015adc18
            not_equal int 0c6a89f13cf1d4e5
            less f32 2f4be0e27c4678a5
            less f64 0b616c56ac6f9e65
            less int 2c5d30e6a2cb3465
            less_equal f32 3325699d782f9f78
            less_equal f64 e3643ce929d5c358
            less_equal int d8ffc94881c04425
            fma f32 5ffbff25342b4682
            fma f64 a41fce1d935fa296
            fma int b8b2835591bec6d9
            where f32 ee634f112d2bf35b
            where f64 5e149b28783ff1ad
            where int 1cf159101f48986d
            idiv int 735deb941f72707f
            and int 6df5b0d261586c1d
            or int c56273adbda0dda9
            xor int 2a77981429e09f31
            threefry int d1c6e25aec1e4932
            |})

(* exp2 is exact at integers whose power is a float, log2 at powers of two,
   subnormal ones included. *)
let test_exact_points =
  test "exp2 and log2 at integers and powers of two" (fun () ->
      for k = -149 to 127 do
        let p = Float.ldexp 1. k in
        equal exact32 (bits32 p) (K.f32 "exp2" [| bits32 (Float.of_int k) |]);
        equal exact32 (bits32 (Float.of_int k)) (K.f32 "log2" [| bits32 p |])
      done;
      for k = -1074 to 1023 do
        let p = Float.ldexp 1. k in
        equal value64 p (K.f64 "exp2" [| Float.of_int k |]);
        equal value64 (Float.of_int k) (K.f64 "log2" [| p |])
      done)

(* f64 transcendentals at the goldens' points, each value correctly rounded by
   mpmath with the exact value's remainder past it: uv run
   dev/nx2/test/kernel/gen/kinds.py. Besides the bound in ranks, each kind's
   largest error against the exact value in ulps of the rounded one, a
   fraction, is held to the largest its algorithm last reached. *)

(* |got - (want + lo)| in ulps of want; 0 where want is no finite number. *)
let fraction64 got want lo =
  if not (Float.is_finite want) then 0.
  else
    let _, e = Float.frexp want in
    let ulp = Float.ldexp 1. (max (e - 53) (-1074)) in
    Float.abs (got -. want -. lo) /. ulp


let test_f64_goldens =
  cases
    ~name:(fun (k, _, _) -> k)
    "f64 at the goldens"
    [
      ("exp", 2, 0.802);
      ("exp2", 2, 1.000);
      ("expm1", 2, 0.649);
      ("log", 2, 0.668);
      ("log2", 2, 0.750);
      ("log1p", 2, 0.709);
      ("sin", 2, 0.642);
      ("cos", 2, 0.651);
      ("tan", 2, 0.729);
      ("asin", 2, 0.607);
      ("acos", 2, 0.598);
      ("atan", 2, 0.573);
      ("sinh", 2, 0.708);
      ("cosh", 2, 0.590);
      ("tanh", 2, 0.832);
      ("erf", 2, 0.714);
      ("pow", 2, 1.000);
      ("atan2", 2, 0.946);
    ]
    (fun (kind, bound, ceiling) ->
      let check row =
        let n = Array.length row - 2 in
        let args = Array.sub row 0 n and want = row.(n) and lo = row.(n + 1) in
        let got = K.f64 kind args in
        let wrong_zero =
          got = 0. && want = 0. && Float.sign_bit got <> Float.sign_bit want
        in
        let e = if wrong_zero then max_int else ulps64 got want in
        let at =
          strf "%s %s: got %h, want %h" kind
            (String.concat " " (List.map (strf "%h") (Array.to_list args)))
            got want
        in
        at_most int ~than:bound e ~msg:at;
        at_most float_exact ~than:ceiling
          (thousandths (fraction64 got want lo))
          ~msg:at
      in
      List.iter check (golden kind))

(* Drawn f32 operands *)

let gen32 =
  let open Gen in
  let any = map (fun i -> Int32.to_int i land 0xFFFF_FFFF) int32 in
  let special =
    of_list
      [
        0;
        0x8000_0000;
        0x7F80_0000;
        0xFF80_0000;
        0x7FC0_0000;
        0x7FA0_0001;
        0xFFC0_0123;
        0x0000_0001;
        0x8000_0001;
        0x007F_FFFF;
        0x0080_0000;
        0x7F7F_FFFF;
        0xFF7F_FFFF;
        bits32 1.;
        bits32 (-1.);
        bits32 0.5;
        bits32 (-0.5);
        bits32 1.5;
        bits32 2.;
        bits32 (-2.);
        bits32 3.;
        bits32 (-3.);
        bits32 0x1p23;
        bits32 (-0x1p23 -. 1.);
      ]
  in
  let moderate = map bits32 (float_range (-20.) 20.) in
  let integers =
    map (fun i -> bits32 (Float.of_int i)) (int_range (-300) 300)
  in
  let halves =
    map (fun i -> bits32 (Float.of_int i +. 0.5)) (int_range (-300) 300)
  in
  let near_one = map (fun i -> bits32 1. + i) (int_range (-4096) 4096) in
  frequency
    [
      (4, any);
      (2, special);
      (3, moderate);
      (2, integers);
      (1, halves);
      (2, near_one);
    ]

let cover32 name b =
  cover (name ^ " NaN") (is_nan b);
  cover (name ^ " -0") (b = 0x8000_0000);
  cover (name ^ " infinite") (b land 0x7FFF_FFFF = 0x7F80_0000);
  cover (name ^ " subnormal") (b land 0x7F80_0000 = 0 && b land 0x7F_FFFF <> 0)

(* f32 exact kinds against their meaning, any NaN for a NaN

   A binary32 sum, difference, product, quotient or square root of floats is the
   double one rounded to binary32, since 53 >= 2 * 24 + 2. *)

let round32 = bits32
let nan32 = bits32 Float.nan

(* The binary32 fma: a b + c is exact as s + e by TwoSum on the double product;
   rounded to odd in binary64, then to binary32, it rounds once. *)
let fma32 a b c =
  let x = f a and y = f b and z = f c in
  if not (Float.is_finite x && Float.is_finite y && Float.is_finite z) then
    round32 (Float.fma x y z)
  else
    let p = Sys.opaque_identity (x *. y) in
    let s = p +. z in
    let bb = s -. p in
    let e = p -. (s -. bb) +. (z -. bb) in
    if e = 0. then round32 s
    else
      let t =
        if e > 0. = (s > 0.) then s
        else Float.copy_sign (Float.pred (Float.abs s)) s
      in
      round32 (Int64.float_of_bits (Int64.logor (bits64 t) 1L))

let binary32 =
  [
    ("add", fun a b -> round32 (f a +. f b));
    ("sub", fun a b -> round32 (f a -. f b));
    ("mul", fun a b -> round32 (f a *. f b));
    ("fdiv", fun a b -> round32 (f a /. f b));
    ("mod", fun a b -> round32 (Float.rem (f a) (f b)));
    (* IEEE 754-2019: -0 below +0 *)
    ( "maximum",
      fun a b ->
        if is_nan a || is_nan b then nan32
        else if f a = f b then a land b
        else if f a > f b then a
        else b );
    ( "minimum",
      fun a b ->
        if is_nan a || is_nan b then nan32
        else if f a = f b then a lor b
        else if f a < f b then a
        else b );
    ("equal", fun a b -> round32 (if f a = f b then 1. else 0.));
    ("not_equal", fun a b -> round32 (if f a <> f b then 1. else 0.));
    ("less", fun a b -> round32 (if f a < f b then 1. else 0.));
    ("less_equal", fun a b -> round32 (if f a <= f b then 1. else 0.));
  ]

let test_binary32 =
  List.map
    (fun (kind, meaning) ->
      prop (strf "%s f32" kind) ~count:20000
        Gen.(pair gen32 gen32)
        (fun (a, b) ->
          cover32 "first" a;
          cover32 "second" b;
          let t =
            match kind with
            | "equal" | "not_equal" | "less" | "less_equal" -> exact32
            | _ -> value32
          in
          equal t (meaning a b) (K.f32 kind [| a; b |])))
    binary32

(* Triples whose exact a b + c lies below the least normal 2^emin of a format of
   precision p, as [round] rounds to it: a product near the subnormals with a
   zero or subnormal addend, or a larger one the addend cancels to its rounding
   error. There an fma that rounds twice, or rounds the product before the sum,
   gives other bits or a zero of the other sign. *)
let tiny ~emin ~p round =
  let open Gen in
  let signed =
    let+ neg = bool and+ u = float_range 1. 2. in
    if neg then -.u else u
  in
  let+ ua = signed
  and+ ub = signed
  and+ ea = int_range ((emin / 2) - 20) ((emin / 2) + 20)
  and+ t = int_range (emin - p - 2) (emin + p - 2)
  and+ s = float_range (-1.) 1.
  and+ addend = int_range 0 4 in
  let a = round (Float.ldexp ua ea) and b = round (Float.ldexp ub (t - ea)) in
  let ab = round (a *. b) and sub = round (Float.ldexp s emin) in
  let c =
    match addend with
    | 0 -> 0.
    | 1 -> -0.
    | 2 -> sub
    | 3 -> -.ab
    | _ -> round (sub -. ab)
  in
  (a, b, c)

let tiny32 =
  Gen.map
    (fun (a, b, c) -> (bits32 a, bits32 b, bits32 c))
    (tiny ~emin:(-126) ~p:24 (fun x -> f (bits32 x)))

let test_fma32 =
  List.map
    (fun target ->
      prop (strf "fma f32 rounds once on %s" target) ~count:20000
        Gen.(frequency [ (1, triple gen32 gen32 gen32); (1, tiny32) ])
        (fun (a, b, c) ->
          let r = fma32 a b c in
          cover32 "addend" c;
          cover "subnormal result"
            (r land 0x7F80_0000 = 0 && r land 0x7F_FFFF <> 0);
          cover "signed zero result" (r = 0x8000_0000);
          equal value32 r (K.f32 ~target "fma" [| a; b; c |])))
    (K.targets ())

let test_where32 =
  prop "where f32"
    Gen.(triple gen32 gen32 gen32)
    (fun (c, a, b) ->
      equal exact32 (if f c <> 0. then a else b) (K.f32 "where" [| c; a; b |]))

(* Unary kinds; neg and abs keep a NaN's payload. *)
let unary32 =
  let float_op op a = round32 (op (f a)) in
  [
    ("neg", (fun a -> a lxor 0x8000_0000), true);
    ("abs", (fun a -> a land 0x7FFF_FFFF), true);
    ( "sign",
      float_op (fun x ->
          if Float.is_nan x then x
          else if x > 0. then 1.
          else if x < 0. then -1.
          else 0.),
      false );
    ("sqrt", float_op Float.sqrt, false);
    ("recip", float_op (fun x -> 1. /. x), false);
    ("floor", float_op Float.floor, false);
    ("ceil", float_op Float.ceil, false);
    ("round", float_op Float.round, false);
    ("trunc", float_op Float.trunc, false);
  ]

let test_unary32 =
  List.map
    (fun (kind, meaning, keeps) ->
      prop (strf "%s f32" kind) ~count:20000 gen32 (fun a ->
          cover32 "operand" a;
          equal
            (if keeps then exact32 else value32)
            (meaning a) (K.f32 kind [| a |])))
    unary32

(* f64 exact kinds, against OCaml's binary64 arithmetic, any NaN for a NaN *)

let gen64 =
  let open Gen in
  let any = map Int64.float_of_bits int64 in
  let special =
    of_list
      [
        0.;
        -0.;
        Float.infinity;
        Float.neg_infinity;
        Float.nan;
        Int64.float_of_bits 0x7FF4_0000_0000_0001L;
        Int64.float_of_bits 1L;
        -.Int64.float_of_bits 1L;
        Float.min_float;
        Float.max_float;
        -.Float.max_float;
        1.;
        -1.;
        0.5;
        -0.5;
        1.5;
        2.;
        0x1p52;
        -0x1p52 -. 1.;
      ]
  in
  let moderate = float_range (-1e6) 1e6 in
  let integers = map Float.of_int (int_range (-1000) 1000) in
  let halves = map (fun i -> Float.of_int i +. 0.5) (int_range (-1000) 1000) in
  frequency
    [ (4, any); (2, special); (3, moderate); (2, integers); (1, halves) ]

let test_f64_exact =
  let binary =
    [
      ("add", ( +. ));
      ("sub", ( -. ));
      ("mul", ( *. ));
      ("fdiv", ( /. ));
      ( "maximum",
        fun x y ->
          if Float.is_nan x || Float.is_nan y then Float.nan
          else if x = y then
            Int64.float_of_bits (Int64.logand (bits64 x) (bits64 y))
          else if x > y then x
          else y );
      ( "minimum",
        fun x y ->
          if Float.is_nan x || Float.is_nan y then Float.nan
          else if x = y then
            Int64.float_of_bits (Int64.logor (bits64 x) (bits64 y))
          else if x < y then x
          else y );
    ]
  in
  List.map
    (fun (kind, op) ->
      prop (strf "%s f64" kind) ~count:20000
        Gen.(pair gen64 gen64)
        (fun (a, b) ->
          cover "NaN" (Float.is_nan a || Float.is_nan b);
          equal value64 (op a b) (K.f64 kind [| a; b |])))
    binary
  @ List.map
      (fun target ->
        prop (strf "fma f64 rounds once on %s" target) ~count:20000
          Gen.(
            frequency
              [
                (1, triple gen64 gen64 gen64);
                (1, tiny ~emin:(-1022) ~p:53 Fun.id);
              ])
          (fun (a, b, c) ->
            let r = Float.fma a b c in
            cover "subnormal result"
              (r <> 0. && Float.abs r < Float.min_float);
            cover "signed zero result" (r = 0. && Float.sign_bit r);
            equal value64 r (K.f64 ~target "fma" [| a; b; c |])))
      (K.targets ())
  @ [
      prop "mod f64" ~count:20000
        Gen.(pair gen64 gen64)
        (fun (a, b) -> equal value64 (Float.rem a b) (K.f64 "mod" [| a; b |]));
    ]
  @ List.map
      (fun (kind, op) ->
        prop (strf "%s f64" kind) ~count:20000 gen64 (fun a ->
            cover "NaN" (Float.is_nan a);
            equal value64 (op a) (K.f64 kind [| a |])))
      [
        ("sqrt", Float.sqrt);
        ("floor", Float.floor);
        ("ceil", Float.ceil);
        ("round", Float.round);
        ("trunc", Float.trunc);
        ("recip", fun x -> 1. /. x);
      ]

(* Integer kinds

   Values are int64s holding a type's values: sign-extended from i32,
   zero-extended from u32. Arithmetic is modular; idiv truncates and is 0 by 0;
   mod takes the dividend's sign and is the dividend by 0; the least value by -1
   wraps; a negative power is 0 but of 1 and -1. *)

type ty = { name : string; bits : int; signed : bool }

let types =
  [
    { name = "i32"; bits = 32; signed = true };
    { name = "u32"; bits = 32; signed = false };
    { name = "i64"; bits = 64; signed = true };
    { name = "u64"; bits = 64; signed = false };
  ]

let wrap t x =
  if t.bits = 64 then x
  else if t.signed then Int64.of_int32 (Int64.to_int32 x)
  else Int64.logand x 0xFFFF_FFFFL

let cmp t a b =
  if t.signed then Int64.compare a b else Int64.unsigned_compare a b

let ipow t a e =
  let rec go r b n =
    if n = 0L then r
    else
      let r = if Int64.logand n 1L = 1L then Int64.mul r b else r in
      go r (Int64.mul b b) (Int64.shift_right_logical n 1)
  in
  if t.signed && Int64.compare e 0L < 0 then
    if a = 1L then 1L
    else if a = -1L then if Int64.logand e 1L = 1L then -1L else 1L
    else 0L
  else wrap t (go 1L a (wrap { t with signed = false } e))

let int_meaning t kind args =
  let w = wrap t and b2i c = if c then 1L else 0L in
  let minus_one = w (-1L) in
  match (kind, args) with
  | "neg", [ a ] -> w (Int64.neg a)
  | "abs", [ a ] ->
      if t.signed && Int64.compare a 0L < 0 then w (Int64.neg a) else a
  | "sign", [ a ] ->
      if a = 0L then 0L
      else if t.signed && Int64.compare a 0L < 0 then -1L
      else 1L
  | "recip", [ a ] -> if a = 1L || (t.signed && a = -1L) then a else 0L
  | "add", [ a; b ] -> w (Int64.add a b)
  | "sub", [ a; b ] -> w (Int64.sub a b)
  | "mul", [ a; b ] -> w (Int64.mul a b)
  | "fma", [ a; b; c ] -> w (Int64.add (Int64.mul a b) c)
  | "idiv", [ a; b ] ->
      if b = 0L then 0L
      else if t.signed && b = minus_one then w (Int64.neg a)
      else if t.signed then Int64.div a b
      else Int64.unsigned_div a b
  | "mod", [ a; b ] ->
      if b = 0L then a
      else if t.signed && b = minus_one then 0L
      else if t.signed then Int64.rem a b
      else Int64.unsigned_rem a b
  | "pow", [ a; e ] -> ipow t a e
  | "maximum", [ a; b ] -> if cmp t a b < 0 then b else a
  | "minimum", [ a; b ] -> if cmp t b a < 0 then b else a
  | "and", [ a; b ] -> Int64.logand a b
  | "or", [ a; b ] -> Int64.logor a b
  | "xor", [ a; b ] -> w (Int64.logxor a b)
  | "equal", [ a; b ] -> b2i (a = b)
  | "not_equal", [ a; b ] -> b2i (a <> b)
  | "less", [ a; b ] -> b2i (cmp t a b < 0)
  | "less_equal", [ a; b ] -> b2i (cmp t a b <= 0)
  | "where", [ c; a; b ] -> if c <> 0L then a else b
  | _ -> invalid_arg kind

let gen_int t =
  let open Gen in
  let extremes =
    List.map (wrap t)
      (if t.bits = 64 then
         [ 0L; 1L; -1L; 2L; -2L; Int64.min_int; Int64.max_int ]
       else [ 0L; 1L; -1L; 2L; -2L; 0x8000_0000L; 0x7FFF_FFFFL ])
  in
  frequency
    [
      (3, map (wrap t) int64);
      (2, of_list extremes);
      (2, map (fun i -> wrap t (Int64.of_int i)) (int_range (-40) 40));
    ]

let int_kinds =
  [
    ("neg", 1);
    ("abs", 1);
    ("sign", 1);
    ("recip", 1);
    ("add", 2);
    ("sub", 2);
    ("mul", 2);
    ("fma", 3);
    ("idiv", 2);
    ("mod", 2);
    ("pow", 2);
    ("maximum", 2);
    ("minimum", 2);
    ("and", 2);
    ("or", 2);
    ("xor", 2);
    ("equal", 2);
    ("not_equal", 2);
    ("less", 2);
    ("less_equal", 2);
    ("where", 3);
  ]

let test_ints =
  List.concat_map
    (fun t ->
      List.map
        (fun (kind, arity) ->
          prop (strf "%s %s" kind t.name) ~count:5000
            Gen.(array ~size:(constant arity) (gen_int t))
            (fun args ->
              let l = Array.to_list args in
              cover "an extreme"
                (List.exists (fun a -> a = wrap t Int64.min_int || a = 0L) l);
              equal int64 (int_meaning t kind l) (K.int t.name kind args)))
        int_kinds)
    types

(* Threefry-2x32-20: Random123's known answers, counter and key low word
   first. *)
let test_threefry =
  let pair lo hi = Int64.(logor (shift_left (of_int hi) 32) (of_int lo)) in
  cases
    ~name:(fun (c, k, _) -> strf "counter %016Lx key %016Lx" c k)
    "threefry"
    [
      (pair 0 0, pair 0 0, pair 0x6b200159 0x99ba4efe);
      ( pair 0xffffffff 0xffffffff,
        pair 0xffffffff 0xffffffff,
        pair 0x1cb996fc 0xbb002be7 );
      ( pair 0x243f6a88 0x85a308d3,
        pair 0x13198a2e 0x03707344,
        pair 0xc4923a9c 0x483df7a0 );
    ]
    (fun (counter, key, want) -> equal int64 want (K.threefry counter key))

(* Complex

   c64 and c128 against their meaning: Add, Sub and Neg are their parts' f32
   and f64 kinds, any NaN for a NaN but Neg's; Mul is within 2u, and Fdiv and Recip within
   4u, of the exact result, normwise, where parts lie in [2^-40, 2^40], u the
   part's unit roundoff; a NaN part in an operand of Mul, Fdiv or Recip makes
   both parts NaN; the order is the real part's, then the imaginary part's. *)


(* A part of magnitude in [2^-40, 2^40], either sign. *)
let part =
  let open Gen in
  let* m = float_range 1. 2. in
  let* e = int_range (-40) 40 in
  let+ s = bool in
  Float.ldexp (if s then -.m else m) e

let parts = Gen.(pair part part)

let c64_of (re, im) = [| bits32 (round_f32 re); bits32 (round_f32 im) |]
let pair_of a = (f a.(0), f a.(1))

let c64_call k ops = pair_of (K.c64 k (Array.concat (List.map c64_of ops)))

let c128_call k ops =
  let r = K.c128 k (Array.concat (List.map (fun (a, b) -> [| a; b |]) ops)) in
  (r.(0), r.(1))

let f32_pair (re, im) = (round_f32 re, round_f32 im)

(* Mul, Fdiv and Recip within their bounds, at c64 and c128. *)
let test_complex_bounds =
  let check name bound u call round exact =
    prop (strf "%s within %gu" name bound) ~count:20000
      Gen.(pair parts parts)
      (fun (a, b) ->
        let a = round a and b = round b in
        let z = exact a b in
        at_most float_exact ~than:bound (normwise u (call a b) z))
  in
  [
    check "mul c64" 2. 0x1p-24
      (fun a b -> c64_call "mul" [ a; b ])
      f32_pair exact_mul;
    check "mul c128" 2. 0x1p-53
      (fun a b -> c128_call "mul" [ a; b ])
      Fun.id exact_mul;
    check "fdiv c64" 4. 0x1p-24
      (fun a b -> c64_call "fdiv" [ a; b ])
      f32_pair exact_div;
    check "fdiv c128" 4. 0x1p-53
      (fun a b -> c128_call "fdiv" [ a; b ])
      Fun.id exact_div;
    check "recip c64" 4. 0x1p-24
      (fun _ b -> c64_call "recip" [ b ])
      f32_pair
      (fun _ b -> exact_div (1., 0.) b);
    check "recip c128" 4. 0x1p-53
      (fun _ b -> c128_call "recip" [ b ])
      Fun.id
      (fun _ b -> exact_div (1., 0.) b);
  ]

(* Add, Sub and Neg are their parts'; Neg keeps a NaN's payload. *)
let test_complex_parts =
  let two32 k (ar, ai, br, bi) =
    equal (array value32)
      [| K.f32 k [| ar; br |]; K.f32 k [| ai; bi |] |]
      (K.c64 k [| ar; ai; br; bi |])
  in
  let two64 k (ar, ai, br, bi) =
    equal (array value64)
      [| K.f64 k [| ar; br |]; K.f64 k [| ai; bi |] |]
      (K.c128 k [| ar; ai; br; bi |])
  in
  let gen32s = Gen.(pair (pair gen32 gen32) (pair gen32 gen32)) in
  let gen64s = Gen.(pair (pair gen64 gen64) (pair gen64 gen64)) in
  let flat ((a, b), (c, d)) = (a, b, c, d) in
  List.concat_map
    (fun k ->
      [
        prop (strf "%s c64" k) gen32s (fun x -> two32 k (flat x));
        prop (strf "%s c128" k) gen64s (fun x -> two64 k (flat x));
      ])
    [ "add"; "sub" ]
  @ [
      prop "neg c64" Gen.(pair gen32 gen32) (fun (a, b) ->
          equal (array exact32)
            [| K.f32 "neg" [| a |]; K.f32 "neg" [| b |] |]
            (K.c64 "neg" [| a; b |]));
      prop "neg c128" Gen.(pair gen64 gen64) (fun (a, b) ->
          equal (array value64)
            [| K.f64 "neg" [| a |]; K.f64 "neg" [| b |] |]
            (K.c128 "neg" [| a; b |]));
    ]

(* A NaN part in an operand of Mul, Fdiv or Recip makes both parts NaN. *)
let test_complex_nan =
  prop "a NaN part makes both parts of mul, fdiv and recip NaN"
    Gen.(triple parts parts (int_range 0 3))
    (fun ((ar, ai), (br, bi), at) ->
      let ops = [| ar; ai; br; bi |] in
      ops.(at) <- Float.nan;
      let a = (ops.(0), ops.(1)) and b = (ops.(2), ops.(3)) in
      let both (re, im) = Float.is_nan re && Float.is_nan im in
      equal bool true (both (c128_call "mul" [ a; b ]));
      equal bool true (both (c128_call "fdiv" [ a; b ]));
      equal bool true (both (c64_call "mul" [ a; b ]));
      equal bool true (both (c64_call "fdiv" [ a; b ]));
      if at >= 2 then begin
        equal bool true (both (c128_call "recip" [ b ]));
        equal bool true (both (c64_call "recip" [ b ]))
      end)

(* A quotient whose imaginary part is 0 while its real part overflows. *)
let test_complex_smith =
  test "fdiv keeps a zero part beside an infinite one" (fun () ->
      let want = (Float.neg_infinity, 0.) in
      let pp ppf (a, b) = Format.fprintf ppf "%h + %hi" a b in
      let t = Testable.make ~pp ~equal:( = ) in
      equal t want
        (c128_call "fdiv" [ (0., 0x1p-50); (0., -.Float.ldexp 1. (-1074)) ]);
      equal t want (c64_call "fdiv" [ (0., 0x1p-20); (0., -0x1p-149) ]))

(* The order: by the real part, then the imaginary part, -0 below +0 for the
   extremes; a NaN part ranks as NaN. *)
let test_complex_order =
  let small = Gen.of_list [ -1.; -0.; 0.; 1.; Float.infinity; Float.nan ] in
  let z = Gen.pair small small in
  let nan (a, b) = Float.is_nan a || Float.is_nan b in
  let cmp (ar, ai) (br, bi) =
    let one x y =
      if x = y then compare (Float.sign_bit y) (Float.sign_bit x)
      else compare x y
    in
    let c = one ar br in
    if c <> 0 then c else one ai bi
  in
  let meaning k a b =
    let n = nan a || nan b in
    let flag p = if p then (1., 0.) else (0., 0.) in
    match k with
    | "equal" -> flag (fst a = fst b && snd a = snd b)
    | "not_equal" -> flag (not (fst a = fst b && snd a = snd b))
    | "less" -> flag ((not n) && (fst a < fst b || (fst a = fst b && snd a < snd b)))
    | "less_equal" ->
        flag ((not n) && (fst a < fst b || (fst a = fst b && snd a <= snd b)))
    | "maximum" -> if nan a then a else if nan b then b else if cmp a b >= 0 then a else b
    | _ -> if nan a then a else if nan b then b else if cmp a b <= 0 then a else b
  in
  (* Each part's bits, any NaN for a NaN: the values travel through f32. *)
  let t = pair value64 value64 in
  List.map
    (fun k ->
      prop (strf "%s by real, then imaginary part" k) Gen.(pair z z)
        (fun (a, b) ->
          cover "NaN" (nan a || nan b);
          cover "equal reals" (fst a = fst b);
          equal t (meaning k a b) (c128_call k [ a; b ]);
          equal t (meaning k a b) (c64_call k [ a; b ])))
    [ "equal"; "not_equal"; "less"; "less_equal"; "maximum"; "minimum" ]

(* The complex transcendental kinds *)

let complex_kinds =
  [
    "exp"; "exp2"; "log"; "log2"; "sqrt"; "sign"; "sin"; "cos"; "tan"; "asin";
    "acos"; "atan"; "sinh"; "cosh"; "tanh"; "asinh"; "acosh"; "atanh";
  ]

(* Each kind's bound, normwise in units of its part's u, and the largest
   error its algorithm last reached at the goldens, at c128 and at c64: a
   change may lower one, never raise it. pow's bound is (3 + 3|w log z|)u. *)
let complex_bounds =
  [
    ("exp", 4., 2.131, 1.779);
    ("exp2", 4., 1.901, 1.922);
    ("log", 4., 1.590, 1.547);
    ("log2", 4., 1.684, 1.581);
    ("sqrt", 3., 1.596, 1.534);
    ("sign", 3., 1.513, 1.407);
    ("asin", 5., 2.620, 2.522);
    ("acos", 5., 2.707, 1.974);
    ("atan", 5., 2.420, 1.907);
    ("sin", 6., 2.586, 3.033);
    ("cos", 6., 2.361, 2.007);
    ("tan", 6., 3.671, 3.009);
    ("sinh", 6., 3.007, 3.020);
    ("cosh", 6., 3.628, 2.124);
    ("tanh", 6., 3.344, 3.053);
    ("asinh", 5., 3.349, 2.840);
    ("acosh", 5., 1.902, 2.278);
    ("atanh", 5., 2.051, 2.454);
  ]


(* |z' - z| / |z| of the moduli alone, in units of [u]. *)
let modulus_error u (re, im) (((zr, _), (zi, _)) : (float * float) * (float * float)) =
  let m = Float.hypot zr zi in
  Float.abs (Float.hypot re im -. m) /. m /. u

let test_complex_goldens =
  let check kind ~bound ~ceiling64 ~ceiling32 =
    let worst128 = ref 0. and worst64 = ref 0. in
    List.iter
      (fun row ->
        let n = Array.length row - 4 in
        let args = Array.sub row 0 n and want = golden_value row n in
        let at got =
          strf "%s %s: got %h %h" kind
            (String.concat " " (List.map (strf "%h") (Array.to_list args)))
            (fst got) (snd got)
        in
        let ops =
          List.init (n / 2) (fun i -> (args.(2 * i), args.(2 * i + 1)))
        in
        let got = c128_call kind ops in
        let e = normwise 0x1p-53 got want in
        at_most float_exact ~than:bound e ~msg:(at got);
        worst128 := Float.max !worst128 e;
        if single_row args want then begin
          let got = c64_call kind ops in
          let e = normwise 0x1p-24 got want in
          at_most float_exact ~than:bound e ~msg:(at got);
          worst64 := Float.max !worst64 e
        end)
      (complex_golden kind);
    at_most float_exact ~than:ceiling64 (thousandths !worst128)
      ~msg:(kind ^ " c128");
    at_most float_exact ~than:ceiling32 (thousandths !worst64)
      ~msg:(kind ^ " c64")
  in
  cases
    ~name:(fun (k, _, _, _) -> k)
    "complex at the goldens" complex_bounds
    (fun (kind, bound, ceiling64, ceiling32) ->
      check kind ~bound ~ceiling64 ~ceiling32)

(* pow within (3 + 3|w log z|)u: its error is log z's, multiplied by w and
   carried into the exponential. *)
let test_complex_pow_goldens =
  test "pow at the goldens" (fun () ->
      List.iter
        (fun row ->
          let args = Array.sub row 0 4 and want = golden_value row 4 in
          let z = (args.(0), args.(1)) and w = (args.(2), args.(3)) in
          let modulus (re, im) = Float.hypot re im in
          let log_z = (Float.log (modulus z), Float.atan2 (snd z) (fst z)) in
          let bound = 3. +. (3. *. modulus w *. modulus log_z) in
          let at (re, im) =
            strf "pow %h %h %h %h: got %h %h" args.(0) args.(1) args.(2)
              args.(3) re im
          in
          let check u got =
            at_most float_exact ~than:bound (normwise u got want) ~msg:(at got)
          in
          check 0x1p-53 (c128_call "pow" [ z; w ]);
          if single_row args want then check 0x1p-24 (c64_call "pow" [ z; w ]))
        (complex_golden "pow"))

(* C99 Annex G's special values, G.6.1 to G.6.4, at c128 and c64. Where it
   leaves a sign unspecified, the value is nx_kinds.h's choice. A NaN part is
   any NaN. *)

let inf = Float.infinity
let ninf = Float.neg_infinity
let nan = Float.nan
let pi = Float.pi

let special_values =
  [
    ("exp", (0., 0.), (1., 0.)); ("exp", (-0., 0.), (1., 0.));
    ("exp", (0., -0.), (1., -0.)); ("exp", (2., inf), (nan, nan));
    ("exp", (2., nan), (nan, nan)); ("exp", (inf, 0.), (inf, 0.));
    ("exp", (ninf, 1.), (0., 0.)); ("exp", (ninf, 3.), (-0., 0.));
    ("exp", (inf, 1.), (inf, inf)); ("exp", (inf, 3.), (ninf, inf));
    ("exp", (ninf, inf), (0., 0.)); ("exp", (ninf, ninf), (0., -0.));
    ("exp", (inf, inf), (inf, nan)); ("exp", (ninf, nan), (0., 0.));
    ("exp", (inf, nan), (inf, nan)); ("exp", (nan, 0.), (nan, 0.));
    ("exp", (nan, 1.), (nan, nan)); ("exp", (nan, nan), (nan, nan));
    ("exp2", (0., 0.), (1., 0.)); ("exp2", (ninf, 1.), (0., 0.));
    ("exp2", (inf, 1.), (inf, inf)); ("exp2", (nan, 0.), (nan, 0.));
    ("exp2", (2., inf), (nan, nan)); ("exp2", (3., 0.), (8., 0.));
    ("log", (-0., 0.), (ninf, pi)); ("log", (0., 0.), (ninf, 0.));
    ("log", (0., -0.), (ninf, -0.)); ("log", (2., inf), (inf, pi /. 2.));
    ("log", (2., nan), (nan, nan)); ("log", (ninf, 1.), (inf, pi));
    ("log", (inf, 1.), (inf, 0.)); ("log", (ninf, inf), (inf, 3. *. pi /. 4.));
    ("log", (inf, inf), (inf, pi /. 4.)); ("log", (inf, nan), (inf, nan));
    ("log", (ninf, nan), (inf, nan)); ("log", (nan, 1.), (nan, nan));
    ("log", (nan, inf), (inf, nan)); ("log", (nan, nan), (nan, nan));
    ("log2", (0., 0.), (ninf, 0.)); ("log2", (inf, 1.), (inf, 0.));
    ("log2", (nan, inf), (inf, nan)); ("log2", (8., 0.), (3., 0.));
    ("sqrt", (0., 0.), (0., 0.)); ("sqrt", (-0., 0.), (0., 0.));
    ("sqrt", (2., inf), (inf, inf)); ("sqrt", (nan, inf), (inf, inf));
    ("sqrt", (2., nan), (nan, nan)); ("sqrt", (ninf, 1.), (0., inf));
    ("sqrt", (inf, 1.), (inf, 0.)); ("sqrt", (ninf, nan), (nan, inf));
    ("sqrt", (inf, nan), (inf, nan)); ("sqrt", (nan, 1.), (nan, nan));
    ("sqrt", (nan, nan), (nan, nan)); ("sqrt", (-4., 0.), (0., 2.));
    ("sqrt", (-4., -0.), (0., -2.));
    ("sinh", (0., 0.), (0., 0.)); ("sinh", (0., inf), (0., nan));
    ("sinh", (0., nan), (0., nan)); ("sinh", (2., inf), (nan, nan));
    ("sinh", (2., nan), (nan, nan)); ("sinh", (inf, 0.), (inf, 0.));
    ("sinh", (inf, 1.), (inf, inf)); ("sinh", (inf, 3.), (ninf, inf));
    ("sinh", (inf, inf), (inf, nan)); ("sinh", (inf, nan), (inf, nan));
    ("sinh", (nan, 0.), (nan, 0.)); ("sinh", (nan, 1.), (nan, nan));
    ("sinh", (nan, nan), (nan, nan));
    ("cosh", (0., 0.), (1., 0.)); ("cosh", (0., inf), (nan, 0.));
    ("cosh", (0., nan), (nan, 0.)); ("cosh", (2., inf), (nan, nan));
    ("cosh", (2., nan), (nan, nan)); ("cosh", (inf, 0.), (inf, 0.));
    ("cosh", (inf, 1.), (inf, inf)); ("cosh", (inf, 3.), (ninf, inf));
    ("cosh", (inf, inf), (inf, nan)); ("cosh", (inf, nan), (inf, nan));
    ("cosh", (nan, 0.), (nan, 0.)); ("cosh", (nan, 1.), (nan, nan));
    ("cosh", (nan, nan), (nan, nan));
    ("tanh", (0., 0.), (0., 0.)); ("tanh", (2., inf), (nan, nan));
    ("tanh", (2., nan), (nan, nan)); ("tanh", (inf, 1.), (1., 0.));
    ("tanh", (inf, 2.), (1., -0.)); ("tanh", (inf, inf), (1., 0.));
    ("tanh", (inf, nan), (1., 0.)); ("tanh", (nan, 0.), (nan, 0.));
    ("tanh", (nan, 1.), (nan, nan)); ("tanh", (nan, nan), (nan, nan));
    ("acos", (0., 0.), (pi /. 2., -0.)); ("acos", (-0., 0.), (pi /. 2., -0.));
    ("acos", (0., nan), (pi /. 2., nan)); ("acos", (2., inf), (pi /. 2., ninf));
    ("acos", (2., nan), (nan, nan)); ("acos", (ninf, 1.), (pi, ninf));
    ("acos", (inf, 1.), (0., ninf));
    ("acos", (ninf, inf), (3. *. pi /. 4., ninf));
    ("acos", (inf, inf), (pi /. 4., ninf)); ("acos", (inf, nan), (nan, ninf));
    ("acos", (nan, 1.), (nan, nan)); ("acos", (nan, inf), (nan, ninf));
    ("acos", (nan, nan), (nan, nan));
    ("asin", (0., 0.), (0., 0.)); ("asin", (-0., 0.), (-0., 0.));
    ("asin", (0., nan), (0., nan)); ("asin", (2., nan), (nan, nan));
    ("asin", (inf, 1.), (pi /. 2., inf)); ("asin", (1., inf), (0., inf));
    ("asin", (inf, inf), (pi /. 4., inf)); ("asin", (nan, inf), (nan, inf));
    ("asin", (inf, nan), (nan, inf)); ("asin", (nan, 1.), (nan, nan));
    ("asin", (nan, 0.), (nan, nan)); ("asin", (nan, nan), (nan, nan));
    ("atan", (0., 0.), (0., 0.)); ("atan", (nan, 0.), (nan, 0.));
    ("atan", (inf, 1.), (pi /. 2., 0.)); ("atan", (1., inf), (pi /. 2., 0.));
    ("atan", (0., 1.), (0., inf)); ("atan", (inf, nan), (pi /. 2., 0.));
    ("atan", (nan, inf), (nan, 0.)); ("atan", (2., nan), (nan, nan));
    ("atan", (nan, 2.), (nan, nan)); ("atan", (nan, nan), (nan, nan));
    ("sin", (0., 0.), (0., 0.)); ("sin", (inf, 0.), (nan, 0.));
    ("cos", (0., 0.), (1., -0.)); ("cos", (inf, 0.), (nan, -0.));
    ("tan", (0., 0.), (0., 0.)); ("tan", (0., inf), (0., 1.));
    ("sign", (inf, 2.), (1., 0.));
    ("sign", (inf, ninf), (sqrt 0.5, -.sqrt 0.5));
    ("sign", (0., -0.), (0., -0.)); ("sign", (nan, 1.), (nan, nan));
    ("sign", (2., nan), (nan, nan)); ("sign", (-3., 4.), (-0.6, 0.8));
    ("asinh", (0., 0.), (0., 0.)); ("asinh", (1., inf), (inf, pi /. 2.));
    ("asinh", (1., nan), (nan, nan)); ("asinh", (inf, 1.), (inf, 0.));
    ("asinh", (inf, inf), (inf, pi /. 4.)); ("asinh", (inf, nan), (inf, nan));
    ("asinh", (nan, 0.), (nan, 0.)); ("asinh", (nan, 1.), (nan, nan));
    ("asinh", (nan, inf), (inf, nan)); ("asinh", (nan, nan), (nan, nan));
    ("acosh", (0., 0.), (0., pi /. 2.)); ("acosh", (-0., 0.), (0., pi /. 2.));
    ("acosh", (2., inf), (inf, pi /. 2.)); ("acosh", (2., nan), (nan, nan));
    ("acosh", (ninf, 1.), (inf, pi)); ("acosh", (inf, 1.), (inf, 0.));
    ("acosh", (ninf, inf), (inf, 3. *. pi /. 4.));
    ("acosh", (inf, inf), (inf, pi /. 4.)); ("acosh", (inf, nan), (inf, nan));
    ("acosh", (ninf, nan), (inf, nan)); ("acosh", (nan, 1.), (nan, nan));
    ("acosh", (nan, inf), (inf, nan)); ("acosh", (nan, nan), (nan, nan));
    ("atanh", (0., 0.), (0., 0.)); ("atanh", (0., nan), (0., nan));
    ("atanh", (1., 0.), (inf, 0.)); ("atanh", (2., inf), (0., pi /. 2.));
    ("atanh", (2., nan), (nan, nan)); ("atanh", (inf, 1.), (0., pi /. 2.));
    ("atanh", (inf, inf), (0., pi /. 2.)); ("atanh", (inf, nan), (0., nan));
    ("atanh", (nan, 1.), (nan, nan)); ("atanh", (nan, inf), (0., pi /. 2.));
    ("atanh", (nan, nan), (nan, nan));
  ]

let test_complex_special =
  cases
    ~name:(fun (k, (x, y), _) -> strf "%s %h %h" k x y)
    "Annex G's special values" special_values
    (fun (kind, z, (re, im)) ->
      let t = pair value64 value64 in
      equal t (re, im) (c128_call kind [ z ]);
      equal t (round_f32 re, round_f32 im) (c64_call kind [ f32_pair z ]))

(* A signalling NaN part beside an infinity or a part past ASIN_BIG: the
   values of a quiet NaN of its sign. A compiler may fold asin's and acos's
   test of both parts against ASIN_BIG into a max, which a signalling NaN
   turns to NaN. *)

let snan = Int64.float_of_bits 0x7FF0_0000_0000_0001L
let nsnan = Int64.float_of_bits 0xFFF0_0000_0000_0001L
let far = 0x1p40

let signalling_values =
  [
    ("asin", (inf, snan), (nan, inf)); ("asin", (ninf, snan), (nan, inf));
    ("asin", (inf, nsnan), (nan, ninf)); ("asin", (snan, inf), (nan, inf));
    ("asin", (snan, ninf), (nan, ninf)); ("asin", (nsnan, inf), (nan, inf));
    ("asin", (far, snan), (nan, nan)); ("asin", (snan, far), (nan, nan));
    ("acos", (inf, snan), (nan, ninf)); ("acos", (ninf, snan), (nan, ninf));
    ("acos", (inf, nsnan), (nan, inf)); ("acos", (snan, inf), (nan, ninf));
    ("acos", (snan, ninf), (nan, inf)); ("acos", (far, snan), (nan, nan));
    ("acos", (snan, far), (nan, nan));
    ("asinh", (inf, snan), (inf, nan)); ("asinh", (ninf, snan), (ninf, nan));
    ("asinh", (inf, nsnan), (inf, nan)); ("asinh", (snan, inf), (inf, nan));
    ("asinh", (nsnan, inf), (ninf, nan)); ("asinh", (snan, ninf), (inf, nan));
    ("asinh", (far, snan), (nan, nan)); ("asinh", (snan, far), (nan, nan));
    ("acosh", (inf, snan), (inf, nan)); ("acosh", (ninf, snan), (inf, nan));
    ("acosh", (inf, nsnan), (inf, nan)); ("acosh", (snan, inf), (inf, nan));
    ("acosh", (snan, ninf), (inf, nan)); ("acosh", (far, snan), (nan, nan));
    ("acosh", (snan, far), (nan, nan));
  ]

(* A part's binary32 bits, a signalling NaN kept signalling. *)
let signalling32 x =
  if Int64.equal (bits64 x) (bits64 snan) then 0x7F80_0001
  else if Int64.equal (bits64 x) (bits64 nsnan) then 0xFF80_0001
  else bits32 (round_f32 x)

let test_complex_signalling =
  cases
    ~name:(fun (k, (x, y), _) ->
      strf "%s 0x%016Lx 0x%016Lx" k (bits64 x) (bits64 y))
    "signalling NaN parts" signalling_values
    (fun (kind, ((x, y) as z), (re, im)) ->
      let t = pair value64 value64 in
      equal t (re, im) (c128_call kind [ z ]);
      equal t (round_f32 re, round_f32 im)
        (pair_of (K.c64 kind [| signalling32 x; signalling32 y |])))

(* Laws over drawn parts, NaNs, infinities, zeros and subnormals among
   them. Each holds bit for bit, any NaN for a NaN. *)

let gen_z64 = Gen.pair gen64 gen64
let gen_z32 = Gen.map (fun (a, b) -> (f a, f b)) (Gen.pair gen32 gen32)
let conj (re, im) = (re, Float.neg im)
let negz (re, im) = (Float.neg re, Float.neg im)
let both_parts = pair value64 value64

(* For each kind, f(conj z) = conj f(z), C99's G.6's symmetry. *)
let test_complex_conj =
  List.concat_map
    (fun k ->
      [
        prop (strf "%s c128 of conj z is its conj" k) gen_z64 (fun z ->
            equal both_parts
              (conj (c128_call k [ z ]))
              (c128_call k [ conj z ]));
        prop (strf "%s c64 of conj z is its conj" k) gen_z32 (fun z ->
            equal both_parts (conj (c64_call k [ z ])) (c64_call k [ conj z ]));
      ])
    complex_kinds

(* The odd kinds give -f(z) at -z, the even ones f(z). *)
let test_complex_parity =
  let odd =
    [ "sinh"; "tanh"; "sin"; "tan"; "asin"; "atan"; "asinh"; "atanh"; "sign" ]
  in
  let even = [ "cosh"; "cos" ] in
  List.concat_map
    (fun k ->
      let side v = if List.mem k odd then negz v else v in
      [
        prop (strf "%s c128 at -z" k) gen_z64 (fun z ->
            equal both_parts
              (side (c128_call k [ z ]))
              (c128_call k [ negz z ]));
        prop (strf "%s c64 at -z" k) gen_z32 (fun z ->
            equal both_parts (side (c64_call k [ z ])) (c64_call k [ negz z ]));
      ])
    (odd @ even)

(* On the real axis, where the float kind is defined, the real part is the
   float kind's value. *)
let test_complex_real_axis =
  let unit64 =
    let ends = Gen.of_list [ 0.; -0.; 1.; -1.; 0x1p-1074 ] in
    Gen.(frequency [ (3, float_range (-1.) 1.); (1, ends) ])
  in
  let unit32 = Gen.map bits32 unit64 in
  let positive64 = Gen.map Float.abs gen64 in
  let positive32 = Gen.map (fun b -> b land 0x7FFF_FFFF) gen32 in
  let real =
    [
      ("exp", gen64, gen32); ("exp2", gen64, gen32); ("sinh", gen64, gen32);
      ("cosh", gen64, gen32); ("tanh", gen64, gen32); ("sin", gen64, gen32);
      ("cos", gen64, gen32); ("tan", gen64, gen32); ("atan", gen64, gen32);
      ("sqrt", positive64, positive32); ("asin", unit64, unit32);
      ("acos", unit64, unit32);
    ]
  in
  List.concat_map
    (fun (k, g64, g32) ->
      [
        prop (strf "%s c128 at x + 0i is the float kind's" k) g64 (fun x ->
            equal value64 (K.f64 k [| x |]) (fst (c128_call k [ (x, 0.) ])));
        prop (strf "%s c64 at x + 0i is the float kind's" k) g32 (fun x ->
            equal value64
              (f (K.f32 k [| x |]))
              (fst (c64_call k [ (f x, 0.) ])));
      ])
    real
  @ List.map
      (fun k ->
        prop (strf "%s c128 at x + 0i is the float kind's of |x|" k) gen64
          (fun x ->
            equal value64
              (K.f64 k [| Float.abs x |])
              (fst (c128_call k [ (x, 0.) ]))))
      [ "log"; "log2" ]
  @ [
      prop "pow c128 of positive reals is the float kind's"
        (Gen.pair gen64 gen64)
        (fun (x, w) ->
          assume (x > 0.);
          equal value64 (K.f64 "pow" [| x; w |])
            (fst (c128_call "pow" [ (x, 0.); (w, 0.) ])));
    ]

(* exp2 at every finite y: |2^(x + iy)| = 2^x for y drawn over the whole
   range of the part's format, and its phase y ln2 within exp2's bound at
   y up to the largest finite part, where it takes y ln2's every bit:
   values from mpmath. *)
let test_complex_exp2_far =
  let anywhere ~emin ~emax =
    let open Gen in
    let* e = int_range emin emax in
    let* m = float_range 1. 1.99 in
    let+ s = bool in
    Float.ldexp (if s then -.m else m) e
  in
  let check name u call round ~emin ~emax ~p =
    prop (strf "|exp2 (x + iy)| is 2^x %s" name)
      Gen.(pair (float_range (-60.) 60.) (anywhere ~emin ~emax))
      (fun (x, y) ->
        let x, y = round (x, y) in
        cover "|y| at least 2^p" (Float.abs y >= Float.ldexp 1. p);
        let want = Float.pow 2. x in
        let got = call "exp2" [ (x, y) ] in
        at_most float_exact ~than:4.
          (modulus_error u got ((want, 0.), (0., 0.))))
  in
  let phase u call y (c, s) =
    let got = call "exp2" [ (0., y) ] in
    at_most float_exact ~than:4. (normwise u got ((c, 0.), (s, 0.)))
      ~msg:(strf "exp2 %h i: got %h %h" y (fst got) (snd got))
  in
  [
    check "c128" 0x1p-53 c128_call Fun.id ~emin:(-1074) ~emax:1023 ~p:53;
    check "c64" 0x1p-24 c64_call f32_pair ~emin:(-149) ~emax:127 ~p:24;
    test "exp2's phase far up the imaginary axis" (fun () ->
        let c128 = phase 0x1p-53 c128_call and c64 = phase 0x1p-24 c64_call in
        c128 0x1p30 (0x1.66e39f669c166p-1, -0x1.6d2939368da98p-1);
        c128 0x1.5eb851eb851ecp+74 (0x1.59f6745a8633cp-1, 0x1.796e36e47cb4ap-1);
        c128 1e300 (0x1.b80528d30f790p-1, -0x1.05c67494c0ab7p-1);
        c128 Float.max_float (0x1.cedaf85d0796ep-1, -0x1.b5c31ba408c70p-2);
        c64 0x1p30 (0x1.66e39f669c166p-1, -0x1.6d2939368da98p-1);
        c64 0x1.5eb852p+74 (-0x1.f0744683c33c2p-1, -0x1.f4d5bd9a1b741p-3);
        c64 0x1.fffffep+127 (0x1.8a07e4d1b0d09p-1, -0x1.46ee442fa86e9p-1));
  ]

(* pow at the values its .mli states: 1 where the exponent is 0, any base
   included; 0 at a zero base where the exponent's real part is positive;
   the sides of the negative real axis by the sign of a zero imaginary
   part. *)
let test_complex_pow_special =
  let one = pair value64 value64 in
  let check call round =
    List.iter
      (fun z -> equal one (1., 0.) (call "pow" [ round z; (0., 0.) ]))
      [ (nan, nan); (0., 0.); (inf, 0.); (2., 3.); (ninf, nan) ];
    equal one (1., 0.) (call "pow" [ round (2., 3.); (-0., 0.) ]);
    List.iter
      (fun w ->
        let re, im = call "pow" [ (0., 0.); round w ] in
        equal (pair float_exact float_exact) (0., 0.) (Float.abs re, Float.abs im))
      [ (2., 0.); (0.5, 0.); (1., 1.); (3., -2.) ]
  in
  [
    test "pow at 0 and 1, c128" (fun () -> check c128_call Fun.id);
    test "pow at 0 and 1, c64" (fun () -> check c64_call f32_pair);
    test "pow on either side of the negative real axis" (fun () ->
        let bound = 3. +. (3. *. 0.5 *. Float.hypot (Float.log 4.) pi) in
        let side u call im want =
          let got = call "pow" [ (-4., im); (0.5, 0.) ] in
          at_most float_exact ~than:bound (normwise u got ((0., 0.), (want, 0.)))
        in
        side 0x1p-53 c128_call 0. 2.;
        side 0x1p-53 c128_call (-0.) (-2.);
        side 0x1p-24 c64_call 0. 2.;
        side 0x1p-24 c64_call (-0.) (-2.));
  ]

(* log |z| near the unit circle keeps its digits: the real part of log at
   points whose modulus is 1 within a few ulps is within 4u of itself. *)
let test_complex_log_circle =
  test "log's real part near the unit circle" (fun () ->
      let check u call z want =
        let re, _ = call "log" [ z ] in
        at_most float_exact ~than:4.
          (Float.abs (re -. want) /. Float.abs want /. u)
          ~msg:(strf "log %h %h: %h" (fst z) (snd z) re)
      in
      check 0x1p-53 c128_call (-0.46684475937821884, -0.8843392848002926)
        0x1.34c78a964b468p-60;
      check 0x1p-53 c128_call (0.6, 0.8) 0x1.999999999999ap-56;
      check 0x1p-53 c128_call (0.8, 0.6) 0x1.999999999999ap-56;
      check 0x1p-53 c128_call (-0.28, 0.96) (-0x1.eb851eb851eb8p-56);
      check 0x1p-24 c64_call (0x1.333334p-1, 0x1.99999ap-1) 0x1.9999995c28f5dp-26;
      check 0x1p-24 c64_call (-0x1.1eb852p-2, 0x1.eb851ep-1)
        (-0x1.5c28f5f65fd8bp-26))

(* Identities within the bounds the kinds' compose to, at parts in
   [2^-40, 2^40]: exp (log z) = z within (5 + 4|log z|)u, the exponential
   carrying log's absolute error into a relative one; sqrt z * sqrt z = z
   within 8u, twice sqrt's bound and mul's. *)
let test_complex_identities =
  let modulus (re, im) = Float.hypot re im in
  let dd z = let re, im = z in ((re, 0.), (im, 0.)) in
  let check name u call round =
    [
      prop (strf "exp (log z) = z %s" name) parts (fun z ->
          let z = round z in
          let l = call "log" [ z ] in
          let bound = 5. +. (4. *. modulus l) in
          at_most float_exact ~than:bound
            (normwise u (call "exp" [ l ]) (dd z)));
      prop (strf "sqrt z * sqrt z = z %s" name) parts (fun z ->
          let z = round z in
          let s = call "sqrt" [ z ] in
          at_most float_exact ~than:8.
            (normwise u (call "mul" [ s; s ]) (dd z)));
    ]
  in
  check "c128" 0x1p-53 c128_call Fun.id @ check "c64" 0x1p-24 c64_call f32_pair

(* C99 Annex F's special values of the float kinds, F.10, and zero results
   of underflow, which keep the exact value's sign: at f64 and f32, bit for
   bit, any NaN for a NaN, through each target's loop and through the best
   one's over a vector, every lane alike. Each value is the f64 result, exact
   or correctly rounded; its rounding to binary32 is the f32 one. *)

let annex_f =
  let pio2 = Float.atan2 1. 0. and pio4 = Float.atan2 1. 1. in
  let tpio4 = Float.atan2 1. (-1.) in
  let one k rows = List.map (fun (x, r) -> (k, [| x |], r)) rows in
  let two k rows = List.map (fun (x, y, r) -> (k, [| x; y |], r)) rows in
  let exp_like k =
    one k [ (0., 1.); (-0., 1.); (ninf, 0.); (inf, inf); (nan, nan) ]
  in
  let log_like k =
    one k
      [
        (0., ninf); (-0., ninf); (1., 0.); (-1., nan); (ninf, nan); (inf, inf);
        (nan, nan);
      ]
  in
  let odd_periodic k =
    one k [ (0., 0.); (-0., -0.); (inf, nan); (ninf, nan); (nan, nan) ]
  in
  List.concat
    [
      exp_like "exp";
      one "exp" [ (-1000., 0.) ];
      exp_like "exp2";
      one "exp2" [ (-2000., 0.) ];
      one "expm1"
        [ (0., 0.); (-0., -0.); (ninf, -1.); (inf, inf); (nan, nan) ];
      log_like "log";
      log_like "log2";
      one "log1p"
        [
          (0., 0.); (-0., -0.); (-1., ninf); (-2., nan); (ninf, nan);
          (inf, inf); (nan, nan);
        ];
      odd_periodic "sin";
      odd_periodic "tan";
      one "cos" [ (0., 1.); (-0., 1.); (inf, nan); (ninf, nan); (nan, nan) ];
      one "asin" [ (0., 0.); (-0., -0.); (2., nan); (-2., nan); (nan, nan) ];
      one "acos" [ (1., 0.); (2., nan); (-2., nan); (nan, nan) ];
      one "atan"
        [ (0., 0.); (-0., -0.); (inf, pio2); (ninf, -.pio2); (nan, nan) ];
      one "sinh"
        [ (0., 0.); (-0., -0.); (inf, inf); (ninf, ninf); (nan, nan) ];
      one "cosh" [ (0., 1.); (-0., 1.); (inf, inf); (ninf, inf); (nan, nan) ];
      one "tanh" [ (0., 0.); (-0., -0.); (inf, 1.); (ninf, -1.); (nan, nan) ];
      one "erf" [ (0., 0.); (-0., -0.); (inf, 1.); (ninf, -1.); (nan, nan) ];
      (* atan2 y x *)
      two "atan2"
        [
          (0., -0., pi); (-0., -0., -.pi); (0., 0., 0.); (-0., 0., -0.);
          (0., -1., pi); (-0., -1., -.pi); (0., 1., 0.); (-0., 1., -0.);
          (-1., 0., -.pio2); (-1., -0., -.pio2); (1., 0., pio2);
          (1., -0., pio2); (1., ninf, pi); (-1., ninf, -.pi); (1., inf, 0.);
          (-1., inf, -0.); (inf, 1., pio2); (ninf, 1., -.pio2);
          (inf, ninf, tpio4); (ninf, ninf, -.tpio4); (inf, inf, pio4);
          (ninf, inf, -.pio4); (nan, 1., nan); (1., nan, nan);
        ];
      two "pow"
        [
          (-0., -3., ninf); (0., -3., inf); (0., ninf, inf); (-0., ninf, inf);
          (0., -2., inf); (-0., -2., inf); (-0., -0.5, inf); (-0., 3., -0.);
          (0., 3., 0.); (-0., 2., 0.); (-0., 0.5, 0.); (-1., inf, 1.);
          (-1., ninf, 1.); (1., nan, 1.); (1., ninf, 1.); (nan, 0., 1.);
          (nan, -0., 1.); (inf, 0., 1.); (-2., 0.5, nan); (0.5, ninf, inf);
          (-0.5, ninf, inf); (2., ninf, 0.); (-2., ninf, 0.); (0.5, inf, 0.);
          (2., inf, inf); (ninf, -3., -0.); (ninf, -2., 0.); (ninf, -0.5, 0.);
          (ninf, 3., ninf); (ninf, 2., inf); (ninf, 0.5, inf); (inf, -1., 0.);
          (inf, 0.5, inf); (nan, 1., nan); (2., nan, nan);
          (-0x1p-60, 3., -0x1p-180); (-0x1p-600, 3., -0.);
        ];
    ]

let test_annex_f =
  let name (k, args, _) =
    strf "%s %s" k
      (String.concat " " (Array.to_list (Array.map (strf "%h") args)))
  in
  let lanes (type s) (k : (float, s) Bigarray.kind) kind args =
    let fill v = Bigarray.Array1.init k Bigarray.c_layout 37 (fun _ -> v) in
    let y = fill 0. in
    let z = if Array.length args > 1 then args.(1) else 0. in
    K.run kind (fill args.(0)) (fill z) y;
    List.init 37 (fun i -> y.{i})
  in
  cases ~name "Annex F's special values" annex_f (fun (kind, args, want) ->
      let args32 = Array.map round_f32 args and want32 = round_f32 want in
      List.iter
        (fun target ->
          equal value64 want (K.f64 ~target kind args) ~msg:target;
          equal value64 want32
            (f (K.f32 ~target kind (Array.map bits32 args32)))
            ~msg:target)
        (K.targets ());
      equal (list value64) (List.init 37 (fun _ -> want))
        (lanes Bigarray.float64 kind args);
      equal (list value64) (List.init 37 (fun _ -> want32))
        (lanes Bigarray.float32 kind args32))

let () =
  exit
  @@ run "nx_array.kinds"
       [
         group ~timeout:10. "transcendentals"
           ((test_record :: test_strided)
           @ [
               test_strata;
               test_narrow;
               test_narrow_binary;
               test_exact_points;
               test_f64_goldens;
               test_annex_f;
               test_digests;
             ]
           @ test_binary_transcendentals);
         group ~timeout:10. "f32"
           (test_binary32 @ test_fma32 @ [ test_where32 ] @ test_unary32);
         group ~timeout:10. "f64" test_f64_exact;
         group ~timeout:10. "integers" test_ints;
         group ~timeout:10. "threefry" [ test_threefry ];
         group ~timeout:10. "complex"
           (test_complex_bounds @ test_complex_parts @ test_complex_order
           @ [ test_complex_nan; test_complex_smith ]);
         group ~timeout:10. "complex transcendentals"
           ([
              test_complex_goldens; test_complex_pow_goldens;
              test_complex_special; test_complex_signalling;
            ]
           @ test_complex_exp2_far @ test_complex_pow_special
           @ [ test_complex_log_circle ]
           @ test_complex_conj @ test_complex_parity @ test_complex_real_axis
           @ test_complex_identities);
       ]
