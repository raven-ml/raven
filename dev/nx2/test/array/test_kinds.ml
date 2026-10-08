(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx_kinds.h: the transcendental kinds within their bounds, f32 at every
   argument and f64 at the goldens' points; the exact kinds against their
   meaning, with NaN, -0, infinities, subnormals and integer extremes drawn. *)

open Windtrap
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

(* Distances in ordered-bit ranks, -0 and +0 one apart; NaN is at no distance
   from NaN and infinitely far from a number. *)

let rank64 x =
  let b = bits64 x in
  if Int64.compare b 0L < 0 then Int64.(sub (neg (logand b max_int)) 1L) else b

let ulps64 a b =
  if Float.is_nan a || Float.is_nan b then
    if Float.is_nan a && Float.is_nan b then 0 else max_int
  else
    let d = Int64.abs (Int64.sub (rank64 a) (rank64 b)) in
    if Int64.compare d (Int64.of_int max_int) >= 0 then max_int
    else Int64.to_int d

let ulps_text e = if e = 1 then "max 1 ulp" else strf "max %d ulps" e

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
   header's. *)

let worst_error (w : K.worst) =
  if Array.length w.errors = 0 then 0 else fst w.errors.(0)

let at_bound kind bound (w : K.worst) =
  let e = worst_error w in
  let p = if e = 0 then 0 else snd w.errors.(0) in
  at_most int ~than:bound e ~msg:(strf "%s at %h (0x%08x)" kind (f p) p)

let test_record =
  test "the full sweep is the headers'" (fun () ->
      let digest, sweeps = K.read_record () in
      if digest <> K.header_digest () then
        failf
          "nx_kinds.h or nx_kinds_real.h changed since the full sweep: run the \
           full sweep (gen/sweep_kinds.exe, from dev/nx2/test/array)";
      equal (list string)
        (List.map fst K.f32_bounds)
        (List.map (fun (k, _, _) -> k) sweeps);
      List.iter
        (fun (kind, e, ps) ->
          let bound = List.assoc kind K.f32_bounds in
          at_most int ~than:bound e ~msg:kind;
          at_bound kind bound (K.points kind (Array.of_list ps)))
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
   ulp of the exact value rounded once. *)
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

(* What every library computing a kind gives, as a digest of its results over
   the strata: the same on each target this CPU runs. The CUDA and Metal
   libraries compute the same strata and compare with these. *)
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
            exp f32 177eda006ee74554
            exp f64 6a6a88d08d90b99f
            exp2 f32 711c6196da734833
            exp2 f64 42bee4b4c44277bb
            expm1 f32 ef4c2c6a23ac8b5e
            expm1 f64 be11e2d19cee6a6b
            log f32 b6731f21538ed139
            log f64 7ad8a48090a38bcd
            log2 f32 778702e38db0be07
            log2 f64 69a393cf9bd4e04c
            log1p f32 dd6eb5095ad8b623
            log1p f64 b0082181f98a7605
            sin f32 2400b7b1a9d8e2bb
            sin f64 e748bf9bce701f07
            cos f32 26df02a72fd3009a
            cos f64 812e9793a43e34cc
            tan f32 40f6d306f2794faf
            tan f64 4a460179a408bd2d
            asin f32 859a2a3112313f95
            asin f64 968bdd52a23a1889
            acos f32 94ad67aa6d92a664
            acos f64 644315a8d014422b
            atan f32 346a9291d387369e
            atan f64 d9ac06d82e275351
            sinh f32 d1c0deff8de95050
            sinh f64 9d23cad5b2079257
            cosh f32 785d82efeb2a2b50
            cosh f64 6eb4470813a62e4a
            tanh f32 9903a9f26ee800c5
            tanh f64 1658d47e32dc44eb
            erf f32 9a8c7a0b844908aa
            erf f64 a9678cf3824af28e
            sqrt f32 4b050dff129930bf
            sqrt f64 560a3d1fea93fab8
            floor f32 cb26e06023a0765c
            floor f64 1c7b6b00a4ebd952
            ceil f32 9414893bf6c3ca89
            ceil f64 23ee0a72743ecfce
            round f32 457cd25d5d947f75
            round f64 528124ef3031db28
            trunc f32 16ba0ab52d914cae
            trunc f64 85c3ef525e379f98
            fdiv f32 270e1959540ee849
            fdiv f64 5bc1dfc2ad6f3ca0
            atan2 f32 832eda2f38c27dc5
            atan2 f64 a5de640dbc768550
            neg f32 ed5e58022ec5c212
            neg f64 82d35d6306e2960a
            neg int 51a191f53249a86d
            abs f32 252f21db1bc3d092
            abs f64 ce93d493e1c8fd8a
            abs int b38dd2f56f9d9e5a
            sign f32 46a41d4d8f721dc5
            sign f64 48133299bfabe105
            sign int 5dea5e3027706bbc
            recip f32 522fe7dd378dad09
            recip f64 c0d07aa072e7f1d5
            recip int b301af41e863f911
            add f32 43803b58f1e555b6
            add f64 ed7c7742a854d805
            add int e43acf1b1b03de31
            sub f32 1236dc8ab5884b76
            sub f64 f1bf7a8d5c71ff59
            sub int 428a370b690eeda1
            mul f32 1124785dde357d0c
            mul f64 e8777f93e6c45d7c
            mul int f6e0a9633e92af59
            mod f32 c4d231ae32d735a8
            mod f64 37e69403fa7eb4fd
            mod int 8f21a2076f3b02f1
            pow f32 87f36c3805f98ddd
            pow f64 dceedd9e267f3b42
            pow int c21a5c07dfddd9d7
            maximum f32 8d8462c4af9141ef
            maximum f64 c4bae58492288516
            maximum int 31e15067e696aeb0
            minimum f32 5582683e10ec30a4
            minimum f64 37b3f5fd8b7c033b
            minimum int 958d0ad38bd04ad4
            equal f32 803ad2f4668a04f8
            equal f64 940226b225d8cfd8
            equal int 5d0597fa23cf4465
            not_equal f32 05567728a31db838
            not_equal f64 a9e59c8d015adc18
            not_equal int 0c6a89f13cf1d4e5
            less f32 f3338469ff4abb98
            less f64 0048845e11aaf385
            less int 2c5d30e6a2cb3465
            less_equal f32 989a60fc67be57c5
            less_equal f64 ae3fe5dee09bd338
            less_equal int d8ffc94881c04425
            fma f32 6fd30f6591312a9a
            fma f64 d4d4a4917501c8ee
            fma int b8b2835591bec6d9
            where f32 492950da23141346
            where f64 b75ac12f7318eb25
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
   mpmath: uv run dev/nx2/test/array/gen/kinds.py *)

let golden kind =
  let ic = open_in (Filename.concat "golden/kinds" (kind ^ ".golden")) in
  let rows = ref [] in
  (try
     while true do
       rows := input_line ic :: !rows
     done
   with End_of_file -> close_in ic);
  List.rev !rows
  |> List.filter (fun l -> l <> "" && l.[0] <> '#')
  |> List.tl
  |> List.map (fun l ->
      String.split_on_char '\t' l |> List.map float_of_string |> Array.of_list)

let test_f64_goldens =
  cases
    ~name:(fun (k, _, _) -> k)
    "f64 at the goldens"
    [
      ("exp", 2, __POS_OF__ {|max 1 ulp|});
      ("exp2", 2, __POS_OF__ {|max 1 ulp|});
      ("expm1", 2, __POS_OF__ {|max 1 ulp|});
      ("log", 2, __POS_OF__ {|max 1 ulp|});
      ("log2", 2, __POS_OF__ {|max 1 ulp|});
      ("log1p", 2, __POS_OF__ {|max 1 ulp|});
      ("sin", 2, __POS_OF__ {|max 1 ulp|});
      ("cos", 2, __POS_OF__ {|max 1 ulp|});
      ("tan", 2, __POS_OF__ {|max 1 ulp|});
      ("asin", 2, __POS_OF__ {|max 1 ulp|});
      ("acos", 2, __POS_OF__ {|max 1 ulp|});
      ("atan", 2, __POS_OF__ {|max 1 ulp|});
      ("sinh", 2, __POS_OF__ {|max 1 ulp|});
      ("cosh", 2, __POS_OF__ {|max 1 ulp|});
      ("tanh", 2, __POS_OF__ {|max 1 ulp|});
      ("erf", 2, __POS_OF__ {|max 1 ulp|});
      ("pow", 2, __POS_OF__ {|max 1 ulp|});
      ("atan2", 2, __POS_OF__ {|max 1 ulp|});
    ]
    (fun (kind, bound, measured) ->
      let check worst row =
        let n = Array.length row - 1 in
        let args = Array.sub row 0 n and want = row.(n) in
        let got = K.f64 kind args in
        let wrong_zero =
          got = 0. && want = 0. && Float.sign_bit got <> Float.sign_bit want
        in
        let e = if wrong_zero then max_int else ulps64 got want in
        at_most int ~than:bound e
          ~msg:
            (strf "%s %s: got %h, want %h" kind
               (String.concat " " (List.map (strf "%h") (Array.to_list args)))
               got want);
        max worst e
      in
      expect (ulps_text (List.fold_left check 0 (golden kind))) measured)

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

(* A NaN operand's bits are the result's where the kind keeps them; a NaN made
   from numbers is any NaN. *)
let nan_first ns r =
  match List.find_opt is_nan ns with Some n -> n | None -> r

let testable_of ns = if List.exists is_nan ns then exact32 else value32

(* f32 exact kinds against their meaning

   A binary32 sum, difference, product, quotient or square root of floats is the
   double one rounded to binary32, since 53 >= 2 * 24 + 2. *)

let round32 = bits32

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
    ("add", fun a b -> nan_first [ a; b ] (round32 (f a +. f b)));
    ("sub", fun a b -> nan_first [ a; b ] (round32 (f a -. f b)));
    ("mul", fun a b -> nan_first [ a; b ] (round32 (f a *. f b)));
    ("fdiv", fun a b -> nan_first [ a; b ] (round32 (f a /. f b)));
    ("mod", fun a b -> round32 (Float.rem (f a) (f b)));
    (* IEEE 754-2019: -0 below +0 *)
    ( "maximum",
      fun a b ->
        nan_first [ a; b ]
          (if f a = f b then a land b else if f a > f b then a else b) );
    ( "minimum",
      fun a b ->
        nan_first [ a; b ]
          (if f a = f b then a lor b else if f a < f b then a else b) );
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
            | "mod" -> value32
            | _ -> testable_of [ a; b ]
          in
          equal t (meaning a b) (K.f32 kind [| a; b |])))
    binary32

let test_fma32 =
  prop "fma f32 rounds once" ~count:20000
    Gen.(triple gen32 gen32 gen32)
    (fun (a, b, c) ->
      cover32 "addend" c;
      equal
        (testable_of [ a; b; c ])
        (nan_first [ a; b; c ] (fma32 a b c))
        (K.f32 "fma" [| a; b; c |]))

let test_where32 =
  prop "where f32"
    Gen.(triple gen32 gen32 gen32)
    (fun (c, a, b) ->
      equal exact32 (if f c <> 0. then a else b) (K.f32 "where" [| c; a; b |]))

(* Unary kinds; neg, abs and sign keep a NaN's bits. *)
let unary32 =
  let float_op op a = nan_first [ a ] (round32 (op (f a))) in
  [
    ("neg", (fun a -> a lxor 0x8000_0000), true);
    ("abs", (fun a -> a land 0x7FFF_FFFF), true);
    ( "sign",
      float_op (fun x -> if x > 0. then 1. else if x < 0. then -1. else 0.),
      true );
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

(* f64 exact kinds, against OCaml's binary64 arithmetic *)

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

let nan_first64 ns r =
  match List.find_opt Float.is_nan ns with Some n -> n | None -> r

let value64_of ns =
  if List.exists Float.is_nan ns then
    Testable.make ~pp:(Testable.pp value64) ~equal:(fun a b ->
        Int64.equal (bits64 a) (bits64 b))
  else value64

let test_f64_exact =
  let binary =
    [
      ("add", ( +. ));
      ("sub", ( -. ));
      ("mul", ( *. ));
      ("fdiv", ( /. ));
      ( "maximum",
        fun x y ->
          if x = y then Int64.float_of_bits (Int64.logand (bits64 x) (bits64 y))
          else if x > y then x
          else y );
      ( "minimum",
        fun x y ->
          if x = y then Int64.float_of_bits (Int64.logor (bits64 x) (bits64 y))
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
          equal
            (value64_of [ a; b ])
            (nan_first64 [ a; b ] (op a b))
            (K.f64 kind [| a; b |])))
    binary
  @ [
      prop "fma f64 rounds once" ~count:20000
        Gen.(triple gen64 gen64 gen64)
        (fun (a, b, c) ->
          equal
            (value64_of [ a; b; c ])
            (nan_first64 [ a; b; c ] (Float.fma a b c))
            (K.f64 "fma" [| a; b; c |]));
    ]
  @ [
      (* fmod promises a NaN, not which *)
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

let () =
  exit
  @@ run "nx_array.kinds"
       [
         group ~timeout:10. "transcendentals"
           ((test_record :: test_strided)
           @ [
               test_strata;
               test_narrow;
               test_exact_points;
               test_f64_goldens;
               test_digests;
             ]
           @ test_binary_transcendentals);
         group ~timeout:10. "f32"
           (test_binary32 @ [ test_fma32; test_where32 ] @ test_unary32);
         group ~timeout:10. "f64" test_f64_exact;
         group ~timeout:10. "integers" test_ints;
         group ~timeout:10. "threefry" [ test_threefry ];
       ]
