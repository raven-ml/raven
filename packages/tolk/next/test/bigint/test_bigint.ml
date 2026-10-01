open Windtrap
module B = Tolk_next.Bigint

let bigint = Testable.make ~pp:B.pp_print ~equal:B.equal
let two_to k = B.shift_left B.one k

(* Python's decimals of 2**64, 2**128 and 2**799, the weak integer's bound. *)
let p64 = "18446744073709551616"
let p128 = "340282366920938463463374607431768211456"

let p799 =
  "3334007216439927137039925895360628898572379161157954080198128905882018618908816035760716100435777145371464955296716620222944400827059682540181678026165415023047578789757007279231539142955907012364482508067943300990845374018738230645581938688"

(* Generators *)

let decimal =
  let open Gen in
  let* n = int_range 1 250 in
  let+ lead = char_range '1' '9'
  and+ rest = string_of ~size:(constant (n - 1)) (char_range '0' '9')
  and+ negative = bool in
  B.of_string ((if negative then "-" else "") ^ String.make 1 lead ^ rest)

(* A power of two moved by a little, around every limb and word boundary. *)
let near_power =
  let open Gen in
  let+ k = int_range 0 900 and+ d = int_range (-2) 2 and+ negative = bool in
  let n = B.add (two_to k) (B.of_int d) in
  if negative then B.neg n else n

let edges =
  Gen.of_list
    [
      min_int; min_int + 1; -1; 0; 1; max_int - 1; max_int; 1 lsl 30; -1 lsl 31;
    ]

let any =
  Gen.with_pp B.pp_print
    (Gen.frequency
       [
         (3, Gen.map B.of_int Gen.small_int);
         (2, Gen.map B.of_int Gen.int);
         (1, Gen.map B.of_int edges);
         (3, near_power);
         (3, decimal);
       ])

let nonzero = Gen.such_that (fun n -> B.sign n <> 0) any
let pair = Gen.pair any any
let small_k = Gen.int_range 0 300

(* Ints whose sums, or products, are ints. *)
let half_int = Gen.int_range (-(1 lsl 61)) ((1 lsl 61) - 1)
let narrow = Gen.int_range (-(1 lsl 30)) (1 lsl 30)

(* Arithmetic on ints *)

let int_reference =
  group "agree with int arithmetic"
    [
      prop "add and sub" (Gen.pair half_int half_int) (fun (x, y) ->
          equal bigint (B.of_int (x + y)) (B.add (B.of_int x) (B.of_int y));
          equal bigint (B.of_int (x - y)) (B.sub (B.of_int x) (B.of_int y)));
      prop "mul" (Gen.pair narrow narrow) (fun (x, y) ->
          equal bigint (B.of_int (x * y)) (B.mul (B.of_int x) (B.of_int y)));
      prop "division rounds as floats do"
        (Gen.pair narrow (Gen.such_that (( <> ) 0) narrow))
        (fun (x, y) ->
          let q = Float.of_int x /. Float.of_int y
          and a = B.of_int x
          and b = B.of_int y in
          equal int (Float.to_int (Float.trunc q)) (B.to_int (B.div a b));
          equal int (Float.to_int (Float.floor q)) (B.to_int (B.fdiv a b));
          equal int (Float.to_int (Float.ceil q)) (B.to_int (B.cdiv a b));
          equal int
            (x - (y * Float.to_int (Float.trunc q)))
            (B.to_int (B.rem a b)));
      prop "bitwise" (Gen.pair Gen.int Gen.int) (fun (x, y) ->
          let a = B.of_int x and b = B.of_int y in
          equal int (x land y) (B.to_int (B.logand a b));
          equal int (x lor y) (B.to_int (B.logor a b));
          equal int (x lxor y) (B.to_int (B.logxor a b));
          equal int (lnot x) (B.to_int (B.lognot a)));
      prop "shifts"
        (Gen.pair narrow (Gen.int_range 0 31))
        (fun (x, k) ->
          equal int (x lsl k) (B.to_int (B.shift_left (B.of_int x) k));
          equal int (x asr k) (B.to_int (B.shift_right (B.of_int x) k)));
      prop "decimal" Gen.int (fun x ->
          equal string (Int.to_string x) (B.to_string (B.of_int x)));
      prop "compare" (Gen.pair Gen.int Gen.int) (fun (x, y) ->
          equal int (Int.compare x y) (B.compare (B.of_int x) (B.of_int y)));
      prop "hash" Gen.int (fun x ->
          equal int (Hashtbl.hash x) (Hashtbl.hash (B.of_int x)));
    ]

(* Laws at every size *)

let laws =
  group "laws"
    [
      prop "add and sub invert" pair (fun (a, b) ->
          equal bigint a (B.sub (B.add a b) b);
          equal bigint (B.add a b) (B.add b a);
          equal bigint (B.neg (B.sub a b)) (B.sub b a));
      prop "mul distributes over add" (Gen.triple any any any) (fun (a, b, c) ->
          equal bigint (B.mul a (B.add b c)) (B.add (B.mul a b) (B.mul a c));
          equal bigint (B.mul a b) (B.mul b a));
      prop "a value has one representation" pair (fun (a, b) ->
          let a' = B.sub (B.add a b) b in
          is_true (a = a');
          equal int (Hashtbl.hash a) (Hashtbl.hash a'));
      prop "compare is the sign of the difference" pair (fun (a, b) ->
          equal int (B.sign (B.sub a b)) (B.compare a b);
          equal bool (B.compare a b = 0) (B.equal a b);
          equal bool (B.compare a b <= 0) (B.leq a b);
          equal bigint (if B.lt a b then a else b) (B.min a b));
      prop "div and rem" (Gen.pair any nonzero) (fun (a, b) ->
          let q = B.div a b and r = B.rem a b in
          equal bigint a (B.add (B.mul b q) r);
          is_true (B.lt (B.abs r) (B.abs b));
          is_true (B.sign r = 0 || B.sign r = B.sign a));
      prop "fdiv, cdiv and ediv" (Gen.pair any nonzero) (fun (a, b) ->
          let rf = B.sub a (B.mul b (B.fdiv a b))
          and rc = B.sub a (B.mul b (B.cdiv a b))
          and re = B.erem a b in
          is_true (B.lt (B.abs rf) (B.abs b) && B.sign rf * B.sign b >= 0);
          is_true (B.lt (B.abs rc) (B.abs b) && B.sign rc * B.sign b <= 0);
          equal bigint a (B.add (B.mul b (B.ediv a b)) re);
          is_true (B.sign re >= 0 && B.lt re (B.abs b)));
      prop "divisible is a zero remainder" pair (fun (a, b) ->
          equal bool
            (if B.sign b = 0 then B.sign a = 0 else B.sign (B.rem a b) = 0)
            (B.divisible a b);
          is_true (B.divisible (B.mul a b) b));
      prop "shifts multiply and floor-divide by powers of two"
        (Gen.pair any small_k) (fun (a, k) ->
          equal bigint (B.mul a (two_to k)) (B.shift_left a k);
          equal bigint (B.fdiv a (two_to k)) (B.shift_right a k));
      prop "bitwise operations" pair (fun (a, b) ->
          equal bigint (B.pred (B.neg a)) (B.lognot a);
          equal bigint
            (B.lognot (B.logand a b))
            (B.logor (B.lognot a) (B.lognot b));
          equal bigint (B.sub (B.logor a b) (B.logand a b)) (B.logxor a b);
          equal bigint (B.add (B.logor a b) (B.logand a b)) (B.add a b));
      prop "extract"
        (Gen.triple any small_k (Gen.int_range 1 200))
        (fun (a, off, len) ->
          let e = B.extract a off len and s = B.signed_extract a off len in
          equal bigint (B.erem (B.shift_right a off) (two_to len)) e;
          equal bigint (B.erem s (two_to len)) e;
          is_true
            (B.leq (B.neg (two_to (len - 1))) s && B.lt s (two_to (len - 1))));
      prop "numbits and trailing_zeros" nonzero (fun a ->
          let n = B.numbits a and z = B.trailing_zeros a in
          is_true (B.leq (two_to (n - 1)) (B.abs a) && B.lt (B.abs a) (two_to n));
          equal bigint a (B.shift_left (B.shift_right a z) z);
          is_true (B.sign (B.extract a z 1) <> 0));
      prop "popcount" (Gen.pair any small_k) (fun (a, k) ->
          let a = B.abs a in
          equal int
            (B.popcount a + 1)
            (B.popcount (B.logor (B.shift_left a (k + 1)) B.one)));
      prop "gcd divides both" pair (fun (a, b) ->
          let g = B.gcd a b in
          is_true (B.sign g >= 0);
          if B.sign g > 0 then (
            equal bigint B.zero (B.rem a g);
            equal bigint B.zero (B.rem b g);
            equal bigint B.one (B.gcd (B.div a g) (B.div b g))));
      prop "pow adds exponents"
        (Gen.triple any (Gen.int_range 0 6) (Gen.int_range 0 6))
        (fun (b, e0, e1) ->
          equal bigint (B.pow b (e0 + e1)) (B.mul (B.pow b e0) (B.pow b e1)));
      prop "sqrt" any (fun a ->
          let a = B.abs a in
          let s = B.sqrt a in
          is_true (B.leq (B.mul s s) a && B.lt a (B.mul (B.succ s) (B.succ s))));
      prop "decimal round trip" any (fun a ->
          equal bigint a (B.of_string (B.to_string a)));
    ]

(* Conversions *)

(* [x], a finite float, is the nearest float to [n], ties to even. *)
let nearest n x =
  let gap = B.abs (B.sub n (B.of_float x)) and _, e = Float.frexp x in
  if e < 54 then equal bigint B.zero gap
  else
    let half_ulp = two_to (e - 54) in
    is_true (B.leq gap half_ulp);
    if B.equal gap half_ulp then
      equal int64 0L (Int64.logand (Int64.bits_of_float x) 1L)

let conversions =
  group "conversions"
    [
      prop "to_float rounds to nearest" any (fun a ->
          let x = B.to_float a in
          if Float.is_finite x then nearest a x);
      prop "of_float rounds toward zero" Gen.float (fun x ->
          if Float.is_finite x then
            equal float_exact
              (if Float.abs x < 1. then 0. else Float.trunc x)
              (B.to_float (B.of_float x)));
      prop "int64" Gen.int64 (fun x ->
          equal int64 x (B.to_int64 (B.of_int64 x));
          equal int64 x (B.to_int64_unsigned (B.of_int64_unsigned x));
          equal int32 (Int64.to_int32 x)
            (B.to_int32 (B.of_int32 (Int64.to_int32 x)));
          equal int32 (Int64.to_int32 x)
            (B.to_int32_unsigned (B.of_int32_unsigned (Int64.to_int32 x))));
      test "ties round to even" (fun () ->
          let x = Float.ldexp 1. 64 in
          equal float_exact x (B.to_float (B.of_string "18446744073709553664"));
          equal float_exact (Float.succ x)
            (B.to_float (B.of_string "18446744073709553665"));
          equal float_exact Float.infinity (B.to_float (two_to 1024)));
      test "of_float" (fun () ->
          equal bigint (B.of_int (-1)) (B.of_float (-1.9));
          equal bigint (two_to 1023) (B.of_float (Float.ldexp 1. 1023));
          equal bigint (B.neg (two_to 62)) (B.of_float (-0x1p62));
          List.iter
            (fun x -> raises B.Overflow (fun () -> B.of_float x))
            [ Float.nan; Float.infinity; Float.neg_infinity ]);
      test "unsigned ranges" (fun () ->
          equal string p64 (B.to_string (B.succ (B.of_int64_unsigned (-1L))));
          equal bigint (two_to 32) (B.succ (B.of_int32_unsigned (-1l)));
          equal int64 (-1L) (B.to_int64_unsigned (B.pred (two_to 64)));
          raises B.Overflow (fun () -> B.to_int64_unsigned (two_to 64));
          raises B.Overflow (fun () -> B.to_int64_unsigned B.minus_one);
          raises B.Overflow (fun () -> B.to_int32_unsigned (two_to 32)));
      test "int and int64 edges" (fun () ->
          let past_int = B.succ (B.of_int max_int) in
          is_true (B.fits_int (B.of_int min_int));
          is_false (B.fits_int past_int);
          is_false (B.fits_int (B.pred (B.of_int min_int)));
          raises B.Overflow (fun () -> B.to_int past_int);
          is_true (B.fits_int64 (B.pred (two_to 63)));
          is_false (B.fits_int64 (two_to 63));
          is_true (B.fits_int64 (B.neg (two_to 63)));
          raises B.Overflow (fun () -> B.to_int64 (two_to 63));
          raises B.Overflow (fun () -> B.to_int32 (two_to 31)));
      test "of_string" (fun () ->
          equal bigint (B.of_int 255) (B.of_string "0xff");
          equal bigint (B.of_int (-5)) (B.of_string "-0b101");
          equal bigint (B.of_int 511) (B.of_string "+0o777");
          equal bigint (B.of_int 1000) (B.of_string "1_000");
          List.iter
            (fun s -> raises_match Exn.invalid_arg (fun () -> B.of_string s))
            [ ""; "-"; "0x"; "_1"; "12a"; " 1"; "0b2" ]);
    ]

(* Boundaries of the int fast paths *)

let boundaries =
  let m = B.of_int min_int and x = B.of_int max_int in
  cases ~name:fst "results past int"
    [
      ("succ max_int", (B.succ x, "4611686018427387904"));
      ("pred min_int", (B.pred m, "-4611686018427387905"));
      ("neg min_int", (B.neg m, "4611686018427387904"));
      ("abs min_int", (B.abs m, "4611686018427387904"));
      ("min_int / -1", (B.div m B.minus_one, "4611686018427387904"));
      ("fdiv min_int -1", (B.fdiv m B.minus_one, "4611686018427387904"));
      ("2^31 * 2^31", (B.mul (two_to 31) (two_to 31), "4611686018427387904"));
      ("-1 * min_int", (B.mul B.minus_one m, "4611686018427387904"));
      ("1 lsl 62", (two_to 62, "4611686018427387904"));
      ("2^64", (two_to 64, p64));
      ("2^128", (two_to 128, p128));
      ("weak int bound", (two_to 799, p799));
      ( "uint64 squared",
        ( B.mul (B.pred (two_to 64)) (B.pred (two_to 64)),
          "340282366920938463426481119284349108225" ) );
      ("numbits min_int", (B.of_int (B.numbits m), "63"));
      ("sqrt 2^128", (B.sqrt (two_to 128), p64));
      ("gcd min_int 0", (B.gcd m B.zero, "4611686018427387904"));
      ( "2^799 / 7 * 7 + rem",
        ( B.add
            (B.mul (B.div (two_to 799) (B.of_int 7)) (B.of_int 7))
            (B.rem (two_to 799) (B.of_int 7)),
          p799 ) );
    ]
    (fun (_, (n, s)) -> equal string s (B.to_string n))

let errors =
  test "errors" (fun () ->
      let b = two_to 100 in
      List.iter
        (fun f ->
          raises Division_by_zero (fun () -> f b B.zero);
          raises Division_by_zero (fun () -> f B.one B.zero))
        [ B.div; B.rem; B.fdiv; B.cdiv; B.ediv; B.erem ];
      let invalid f = raises_match Exn.invalid_arg f in
      invalid (fun () -> B.shift_left B.one (-1));
      invalid (fun () -> B.shift_right b (-1));
      invalid (fun () -> B.extract b (-1) 3);
      invalid (fun () -> B.signed_extract B.one 0 0);
      invalid (fun () -> B.pow B.one (-1));
      invalid (fun () -> B.sqrt B.minus_one);
      raises B.Overflow (fun () -> B.popcount B.minus_one);
      equal int max_int (B.trailing_zeros B.zero);
      equal int 0 (B.numbits B.zero))

let () =
  exit
    (run "Tolk_next.Bigint"
       [ int_reference; laws; conversions; boundaries; errors ])
