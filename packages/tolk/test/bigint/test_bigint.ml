open Windtrap
module B = Tolk.Bigint

let bigint =
  Testable.with_compare B.compare (Testable.make ~pp:B.pp_print ~equal:B.equal)

let pairs = Testable.pair bigint bigint
let two_to k = B.shift_left B.one k
let power_of_two k = B.pow (B.of_int 2) k
let int_min = B.of_int min_int
let int_max = B.of_int max_int

(* [max_int + 1], from [int]'s width: 2^62 where it has 63 bits. *)
let past_int = two_to (Sys.int_size - 1)
let int64_min = B.neg (two_to 63)
let int64_max = B.pred (two_to 63)

(* Integers from here on round to an infinity (IEEE 754-2019 §4.3.1). *)
let overflow = B.sub (two_to 1024) (two_to 970)
let pp_pair ppf (a, b) = Format.fprintf ppf "(%a, %a)" B.pp_print a B.pp_print b

(* Generators *)

(* The decimal text of an integer of 1 to 250 digits. *)
let long_text =
  let open Gen in
  let* n = int_range 1 250 in
  let+ lead = char_range '1' '9'
  and+ rest = string_of ~size:(constant (n - 1)) (char_range '0' '9')
  and+ negative = bool in
  (if negative then "-" else "") ^ String.make 1 lead ^ rest

let edges =
  Gen.of_list ~pp:Format.pp_print_int
    [
      min_int; min_int + 1; -1; 0; 1; max_int - 1; max_int; 1 lsl 30; -1 lsl 31;
    ]

(* Canonical decimal texts, of ints and of integers past them. *)
let text =
  Gen.with_pp
    (fun ppf -> Format.fprintf ppf "%S")
    (Gen.frequency
       [
         (2, Gen.map Int.to_string Gen.small_int);
         (1, Gen.map Int.to_string edges);
         (2, Gen.map Int.to_string Gen.int);
         (3, long_text);
       ])

(* A power of two moved by a little, around every limb and word boundary and
   past the greatest float. *)
let near_power =
  let open Gen in
  let+ k = int_range 0 1100 and+ d = int_range (-2) 2 and+ negative = bool in
  let n = B.add (two_to k) (B.of_int d) in
  if negative then B.neg n else n

let any =
  Gen.with_pp B.pp_print
    (Gen.frequency
       [
         (3, Gen.map B.of_int Gen.small_int);
         (2, Gen.map B.of_int Gen.int);
         (1, Gen.map B.of_int edges);
         (3, near_power);
         (3, Gen.map B.of_string long_text);
       ])

let nonzero = Gen.such_that (fun n -> B.sign n <> 0) any

(* Pairs, one in five of equal integers built apart. *)
let pair =
  Gen.with_pp pp_pair
    (Gen.frequency
       [
         (4, Gen.pair any any);
         (1, Gen.map (fun a -> (a, B.of_string (B.to_string a))) any);
       ])

(* A dividend and a divisor that is not zero, one in four dividing it. *)
let division =
  Gen.with_pp pp_pair
    (Gen.frequency
       [
         (3, Gen.pair any nonzero);
         (1, Gen.map (fun (q, b) -> (B.mul q b, b)) (Gen.pair any nonzero));
       ])

let small_k = Gen.int_range 0 300

(* Ints whose sums, or products, are ints. *)
let half_int = Gen.int_range (-(1 lsl 61)) ((1 lsl 61) - 1)
let narrow = Gen.int_range (-(1 lsl 30)) (1 lsl 30)

let int_pair g =
  Gen.with_pp
    (fun ppf (x, y) -> Format.fprintf ppf "(%d, %d)" x y)
    (Gen.pair g g)

(* Integers *)

let exact_rows =
  [
    ("succ max_int", (fun () -> B.succ int_max), past_int);
    ("pred min_int", (fun () -> B.pred int_min), B.pred (B.neg past_int));
    ( "add max_int max_int",
      (fun () -> B.add int_max int_max),
      B.sub (B.shift_left past_int 1) (B.of_int 2) );
    ("sub 0 min_int", (fun () -> B.sub B.zero int_min), past_int);
    ("neg min_int", (fun () -> B.neg int_min), past_int);
    ("abs min_int", (fun () -> B.abs int_min), past_int);
    ("mul min_int -1", (fun () -> B.mul int_min B.minus_one), past_int);
    ("mul -1 min_int", (fun () -> B.mul B.minus_one int_min), past_int);
    ( "mul 2^31 2^(int's width - 32)",
      (fun () ->
        B.mul (B.of_int (1 lsl 31)) (B.of_int (1 lsl (Sys.int_size - 32)))),
      past_int );
    ("div min_int -1", (fun () -> B.div int_min B.minus_one), past_int);
    ("fdiv min_int -1", (fun () -> B.fdiv int_min B.minus_one), past_int);
    ("cdiv min_int -1", (fun () -> B.cdiv int_min B.minus_one), past_int);
    ("gcd min_int 0", (fun () -> B.gcd int_min B.zero), past_int);
    ( "shift_left min_int 1",
      (fun () -> B.shift_left int_min 1),
      B.neg (B.shift_left past_int 1) );
  ]

let integers =
  group "integers"
    [
      cases "results past int's range are exact, with nothing wrapped"
        ~id:(fun (id, _, _) -> id)
        exact_rows
        (fun (_, result, expected) -> equal bigint expected (result ()));
      prop
        "each integer has one representation, so polymorphic equality and \
         Hashtbl.hash apply"
        (Gen.pair any any) (fun (a, b) ->
          let a' = B.sub (B.add a b) b in
          equal (Testable.make ~pp:B.pp_print ~equal:Stdlib.( = )) a a';
          equal int (Hashtbl.hash a) (Hashtbl.hash a'));
      prop "an integer that fits an int hashes as that int" Gen.int (fun x ->
          equal int (Hashtbl.hash x) (Hashtbl.hash (B.of_int x)));
    ]

(* Constants *)

let constants =
  group "constants"
    [
      cases "zero, one and minus_one are 0, 1 and -1" ~id:fst
        [
          ("zero", (B.zero, "0"));
          ("one", (B.one, "1"));
          ("minus_one", (B.minus_one, "-1"));
        ]
        (fun (_, (n, decimal)) -> equal string decimal (B.to_string n));
    ]

(* Conversions *)

let in_int n = B.leq int_min n && B.leq n int_max
let in_int64 n = B.leq int64_min n && B.leq n int64_max

(* [x], finite, is the float nearest to [n], ties to even. Floats below 2^53
   hold every integer. Above, [x]'s neighbours are an ulp away, or half one
   below [x] where [x] is a power of two that [n] lies under. *)
let nearest n x =
  let gap = B.abs (B.sub n (B.of_float x)) and m, e = Float.frexp x in
  if e < 54 then equal bigint B.zero gap
  else
    let under = Float.abs m = 0.5 && B.lt (B.abs n) (B.abs (B.of_float x)) in
    let ulp = two_to (if under then e - 54 else e - 53) in
    let twice = B.shift_left gap 1 in
    at_most bigint ~than:ulp twice;
    if B.equal twice ulp then
      satisfies ~claim:"an even significand" float_exact
        (fun x -> Int64.(equal (logand (bits_of_float x) 1L) 0L))
        x

(* Ties between two floats, below and above 2^64 and under 2^60, and the
   integers next to one. *)
let tie_examples =
  let p64 = two_to 64 in
  [
    B.add p64 (two_to 11);
    B.add p64 (B.mul (B.of_int 3) (two_to 11));
    B.neg (B.add p64 (two_to 11));
    B.sub (two_to 60) (two_to 6);
    B.succ (B.add p64 (two_to 11));
    overflow;
    B.pred overflow;
  ]

let to_float_rows =
  let max = B.of_float Float.max_float in
  [
    ("max_float + 1", B.succ max, Float.max_float);
    ("2^1024 - 2^970 - 1", B.pred overflow, Float.max_float);
    ("2^1024 - 2^970", overflow, Float.infinity);
    ("2^1024", two_to 1024, Float.infinity);
    ("-(2^1024 - 2^970 - 1)", B.neg (B.pred overflow), -.Float.max_float);
    ("-(2^1024 - 2^970)", B.neg overflow, Float.neg_infinity);
  ]

let ten_to_4300 () = B.pow (B.of_int 10) 4300

let conversions =
  group "conversions"
    [
      prop "of_int n writes as Int.to_string n" Gen.int (fun x ->
          equal string (Int.to_string x) (B.to_string (B.of_int x)));
      prop "of_int32 n writes as Int32.to_string n" Gen.int32 (fun x ->
          equal string (Int32.to_string x) (B.to_string (B.of_int32 x)));
      prop "of_int64 n writes as Int64.to_string n" Gen.int64 (fun x ->
          equal string (Int64.to_string x) (B.to_string (B.of_int64 x)));
      prop "of_int32_unsigned n writes as %lu writes n" ~examples:[ -1l ]
        Gen.int32 (fun x ->
          equal string (Printf.sprintf "%lu" x)
            (B.to_string (B.of_int32_unsigned x)));
      prop "of_int64_unsigned n writes as %Lu writes n" ~examples:[ -1L ]
        Gen.int64 (fun x ->
          equal string (Printf.sprintf "%Lu" x)
            (B.to_string (B.of_int64_unsigned x)));
      prop "of_float x is x rounded toward zero"
        ~examples:[ -1.9; -0.5; 0.5; -0.; 0x1p62; -0x1p62; 0x1p1023 ] Gen.float
        (fun x ->
          (* [+. 0.] makes [Float.trunc]'s [-0.] the integer zero. *)
          equal float_exact (Float.trunc x +. 0.) (B.to_float (B.of_float x)));
      cases "of_float raises Overflow if x is infinite or NaN"
        ~id:Float.to_string [ Float.infinity; Float.neg_infinity; Float.nan ]
        (fun x -> raises B.Overflow (fun () -> B.of_float x));
      cases
        "of_string reads a sign, a base's prefix in either case, then digits \
         and underscores"
        ~id:fst
        [
          ("0", 0);
          ("-0", 0);
          ("+7", 7);
          ("007", 7);
          ("0xff", 255);
          ("0XFF", 255);
          ("0xFf", 255);
          ("0x1_f", 31);
          ("-0b101", -5);
          ("0B11", 3);
          ("+0o777", 511);
          ("0O17", 15);
          ("1_000", 1000);
          ("1__0", 10);
          ("1_", 1);
        ]
        (fun (s, n) -> equal bigint (B.of_int n) (B.of_string s));
      cases "of_string raises Invalid_argument on any other text"
        ~id:(Printf.sprintf "%S")
        [
          "";
          "-";
          "+";
          "0x";
          "0X";
          "_1";
          "-_1";
          "0x_f";
          "12a";
          " 1";
          "1 ";
          "0b2";
          "0o8";
          "0xg";
          "+-1";
          "--1";
          "1.0";
          "1e3";
          "0u12";
        ] (fun s -> raises_match Exn.invalid_arg (fun () -> B.of_string s));
      prop "of_string reads bases 2, 8 and 16 as their digits' sum"
        Gen.(
          triple
            (of_list
               ~pp:(fun ppf (_, p) -> Format.pp_print_string ppf p)
               [ (2, "0b"); (8, "0o"); (16, "0x") ])
            (pair bool bool)
            (list ~size:(int_range 1 120) (int_range 0 15)))
        (fun ((base, prefix), (negative, upper), ds) ->
          let ds = List.map (fun d -> d mod base) ds in
          let sum =
            List.fold_left
              (fun acc d -> B.add (B.mul acc (B.of_int base)) (B.of_int d))
              B.zero ds
          in
          let digit =
            if upper then Printf.sprintf "%X" else Printf.sprintf "%x"
          in
          let text = String.concat "" (List.map digit ds) in
          let sign = if negative then "-" else "" in
          equal bigint
            (if negative then B.neg sum else sum)
            (B.of_string (sign ^ prefix ^ text)));
      prop "of_string inverts to_string" any
        (Law.round_trip bigint string B.to_string B.of_string);
      test
        "of_string reads 4300 decimal digits, leading zeros included and \
         underscores not" (fun () ->
          let nines = String.make 4300 '9' in
          equal bigint (B.pred (ten_to_4300 ())) (B.of_string nines);
          equal bigint
            (B.neg (B.pred (ten_to_4300 ())))
            (B.of_string ("-" ^ nines));
          equal bigint B.zero (B.of_string (String.make 4300 '0'));
          equal bigint
            (B.pow (B.of_int 10) 4299)
            (B.of_string ("1_" ^ String.make 4299 '0')));
      cases "of_string raises Invalid_argument past 4300 decimal digits"
        ~id:(fun (prefix, digit, count) ->
          Printf.sprintf "%S then %d %C" prefix count digit)
        [
          ("", '9', 4301);
          ("-", '9', 4301);
          ("", '0', 4301);
          ("1_", '9', 4300);
          ("", '9', 2_000_000);
        ]
        (fun (prefix, digit, count) ->
          let s = prefix ^ String.make count digit in
          raises_match Exn.invalid_arg (fun () -> B.of_string s));
      test
        "of_string reads 2,000,000 hexadecimal digits: bases 2, 8 and 16 have \
         no limit" (fun () ->
          equal bigint
            (B.pred (two_to 8_000_000))
            (B.of_string ("0x" ^ String.make 2_000_000 'f')));
      prop "fits_int n is true iff n is in int's range"
        ~examples:[ int_min; int_max; B.pred int_min; B.succ int_max ]
        any
        (fun n ->
          cover "in range" (in_int n);
          cover "out of range" (not (in_int n));
          satisfies ~claim:"fits_int n iff min_int <= n <= max_int" bigint
            (fun n -> Bool.equal (B.fits_int n) (in_int n))
            n);
      prop "to_int n is n in int's range, and raises Overflow out of it"
        ~examples:[ int_min; int_max; B.pred int_min; B.succ int_max ]
        any
        (fun n ->
          cover "in range" (in_int n);
          cover "out of range" (not (in_int n));
          if in_int n then equal bigint n (B.of_int (B.to_int n))
          else raises B.Overflow (fun () -> B.to_int n));
      prop "to_int64 n is n in int64's range, and raises Overflow out of it"
        ~examples:[ int64_min; int64_max; B.pred int64_min; B.succ int64_max ]
        any
        (fun n ->
          cover "in range" (in_int64 n);
          cover "out of range" (not (in_int64 n));
          if in_int64 n then equal bigint n (B.of_int64 (B.to_int64 n))
          else raises B.Overflow (fun () -> B.to_int64 n));
      prop "to_float n is the float nearest to n, ties to even, or an infinity"
        ~examples:tie_examples any (fun n ->
          let x = B.to_float n in
          cover "an infinity" (not (Float.is_finite x));
          cover "inexact" (Float.is_finite x && not (B.equal n (B.of_float x)));
          if Float.is_finite x then nearest n x
          else (
            at_least bigint ~than:overflow (B.abs n);
            equal float_exact
              (if B.sign n < 0 then Float.neg_infinity else Float.infinity)
              x));
      cases "to_float n is an infinity of n's sign iff abs n >= 2^1024 - 2^970"
        ~id:(fun (id, _, _) -> id)
        to_float_rows
        (fun (_, n, x) -> equal float_exact x (B.to_float n));
      cases "to_string writes powers of two and their negations as Python's str"
        ~id:(fun (k, _) -> Printf.sprintf "2^%d" k)
        [
          (* Python's [str(2**k)]. *)
          (62, "4611686018427387904");
          (64, "18446744073709551616");
          (128, "340282366920938463463374607431768211456");
          ( 799,
            "3334007216439927137039925895360628898572379161157954080198128905882018618908816035760716100435777145371464955296716620222944400827059682540181678026165415023047578789757007279231539142955907012364482508067943300990845374018738230645581938688"
          );
        ]
        (fun (k, decimal) ->
          equal string decimal (B.to_string (power_of_two k));
          equal string ("-" ^ decimal) (B.to_string (B.neg (power_of_two k))));
      test "to_string writes 4300 digits" (fun () ->
          let nines = String.make 4300 '9' in
          equal string nines (B.to_string (B.pred (ten_to_4300 ())));
          equal string ("-" ^ nines)
            (B.to_string (B.neg (B.pred (ten_to_4300 ())))));
      cases "to_string raises Invalid_argument past 4300 digits" ~id:fst
        [
          ("10^4300", fun () -> ten_to_4300 ());
          ("-10^4300", fun () -> B.neg (ten_to_4300 ()));
          ("2^9_437_208", fun () -> two_to 9_437_208);
        ]
        (fun (_, n) ->
          let n = n () in
          raises_match Exn.invalid_arg (fun () -> B.to_string n));
      prop "pp_print formats integers as to_string does" any (fun n ->
          equal string (B.to_string n) (Format.asprintf "%a" B.pp_print n));
      test "pp_print raises as to_string does past 4300 digits" (fun () ->
          let n = ten_to_4300 () in
          raises_match Exn.invalid_arg (fun () ->
              Format.asprintf "%a" B.pp_print n));
    ]

(* Predicates and comparisons *)

(* The sign a canonical decimal writes. *)
let text_sign s =
  if s.[0] = '-' then -1 else if String.equal s "0" then 0 else 1

(* The order of two canonical decimals: by sign, then by length, then by digits,
   reversed between negatives. *)
let text_compare s t =
  match (text_sign s, text_sign t) with
  | a, b when a <> b -> Int.compare a b
  | sign, _ ->
      let c =
        match Int.compare (String.length s) (String.length t) with
        | 0 -> Int.compare (String.compare s t) 0
        | c -> c
      in
      if sign < 0 then -c else c

(* Pairs of texts, one in five equal. *)
let text_pair =
  Gen.with_pp
    (fun ppf (s, t) -> Format.fprintf ppf "(%S, %S)" s t)
    (Gen.frequency
       [ (4, Gen.pair text text); (1, Gen.map (fun s -> (s, s)) text) ])

let order =
  group "predicates and comparisons"
    [
      prop "sign n is the sign n's decimal writes" ~examples:[ "0"; "-1"; "1" ]
        text (fun s -> equal int (text_sign s) (B.sign (B.of_string s)));
      prop "compare orders integers as their decimals are ordered"
        ~examples:[ ("-1", "-1"); ("-10", "-9"); ("9", "10") ]
        text_pair
        (fun (s, t) ->
          let expected = text_compare s t in
          cover "less" (expected < 0);
          cover "equal" (expected = 0);
          cover "greater" (expected > 0);
          equal int expected
            (Int.compare (B.compare (B.of_string s) (B.of_string t)) 0));
      prop "equal n0 n1 is compare n0 n1 = 0"
        ~examples:[ (two_to 100, B.of_string (B.to_string (two_to 100))) ]
        pair
        (fun (a, b) ->
          cover "equal" (B.compare a b = 0);
          satisfies ~claim:"equal iff compare is 0" pairs
            (fun (a, b) -> Bool.equal (B.equal a b) (B.compare a b = 0))
            (a, b));
      prop "leq n0 n1 is compare n0 n1 <= 0" pair (fun (a, b) ->
          satisfies ~claim:"leq iff compare <= 0" pairs
            (fun (a, b) -> Bool.equal (B.leq a b) (B.compare a b <= 0))
            (a, b));
      prop "geq n0 n1 is compare n0 n1 >= 0" pair (fun (a, b) ->
          satisfies ~claim:"geq iff compare >= 0" pairs
            (fun (a, b) -> Bool.equal (B.geq a b) (B.compare a b >= 0))
            (a, b));
      prop "lt n0 n1 is compare n0 n1 < 0" pair (fun (a, b) ->
          satisfies ~claim:"lt iff compare < 0" pairs
            (fun (a, b) -> Bool.equal (B.lt a b) (B.compare a b < 0))
            (a, b));
      prop "gt n0 n1 is compare n0 n1 > 0" pair (fun (a, b) ->
          satisfies ~claim:"gt iff compare > 0" pairs
            (fun (a, b) -> Bool.equal (B.gt a b) (B.compare a b > 0))
            (a, b));
      prop "min n0 n1 is the least of n0 and n1" pair (fun (a, b) ->
          equal bigint (if B.compare a b <= 0 then a else b) (B.min a b));
      prop "max n0 n1 is the greatest of n0 and n1" pair (fun (a, b) ->
          equal bigint (if B.compare a b >= 0 then a else b) (B.max a b));
    ]

(* Arithmetic *)

(* The minor words [f ()] allocates, with those of measuring. Kept apart from
   its callers, so that [f ()] stays a call. *)
let[@inline never] [@local never] words f =
  let before = Gc.minor_words () in
  let r = f () in
  let after = Gc.minor_words () in
  ignore (Sys.opaque_identity r);
  after -. before

(* Less those of measuring: in bytecode, reading the counter allocates the float
   it returns. *)
let allocated f = Float.to_int (words f -. words (fun () -> ()))
let small = Gen.frequency [ (1, edges); (2, Gen.int); (2, narrow) ]

let allocates_nothing (x, y) =
  let a = B.of_int x and b = B.of_int y in
  let fits name f =
    match f () with
    | exception Division_by_zero -> ()
    | r -> if B.fits_int r then equal ~msg:name int 0 (allocated f)
  in
  fits "neg" (fun () -> B.neg a);
  fits "abs" (fun () -> B.abs a);
  fits "succ" (fun () -> B.succ a);
  fits "pred" (fun () -> B.pred a);
  fits "add" (fun () -> B.add a b);
  fits "sub" (fun () -> B.sub a b);
  fits "mul" (fun () -> B.mul a b);
  fits "div" (fun () -> B.div a b);
  fits "rem" (fun () -> B.rem a b);
  fits "fdiv" (fun () -> B.fdiv a b);
  fits "cdiv" (fun () -> B.cdiv a b);
  fits "divisible" (fun () -> if B.divisible a b then B.one else B.zero);
  fits "gcd" (fun () -> B.gcd a b);
  fits "pow" (fun () -> B.pow a (y land 7))

(* [s] with its sign flipped, and without it, as canonical decimals write
   them. *)
let negate_text s =
  if s.[0] = '-' then String.sub s 1 (String.length s - 1)
  else if String.equal s "0" then s
  else "-" ^ s

let abs_text s = if s.[0] = '-' then negate_text s else s

(* [q] is [a / b] rounded as [rounding] says: [a - b * q] is smaller than [b] in
   magnitude, and zero or of [a]'s sign ([`Zero]), of [b]'s ([`Down]) or of the
   opposite of [b]'s ([`Up]). *)
let rounds rounding q (a, b) =
  let r = B.sub a (B.mul b q) in
  cover "exact" (B.sign r = 0);
  cover "inexact, of a negative quotient"
    (B.sign r <> 0 && B.sign a * B.sign b < 0);
  cover "inexact, of a positive quotient"
    (B.sign r <> 0 && B.sign a * B.sign b > 0);
  less bigint ~than:(B.abs b) (B.abs r);
  let side, claim =
    match rounding with
    | `Zero -> (B.sign a, "zero or of a's sign")
    | `Down -> (B.sign b, "zero or of b's sign")
    | `Up -> (-B.sign b, "zero or of the opposite of b's sign")
  in
  satisfies ~claim int (fun s -> s = 0 || s = side) (B.sign r)

let division_examples =
  [
    (int_min, B.minus_one);
    (int_min, B.one);
    (two_to 799, B.of_int 7);
    (B.of_int (-7), B.of_int 2);
    (B.of_int 7, B.of_int (-2));
  ]

(* Pairs, one in two the multiples of a common factor. *)
let gcd_pair =
  Gen.with_pp pp_pair
    (Gen.frequency
       [
         (1, Gen.pair any any);
         ( 1,
           Gen.map
             (fun (a, b, c) -> (B.mul c a, B.mul c b))
             (Gen.triple any any any) );
       ])

let arithmetic =
  group "arithmetic"
    [
      prop "allocates nothing while its arguments and result fit an int"
        ~examples:
          [ (min_int, -1); (min_int, 1); (max_int, -1); (7, -2); (-7, 2) ]
        (int_pair small) allocates_nothing;
      prop "neg n is -n, as n's decimal with its sign flipped" text (fun s ->
          equal bigint (B.of_string (negate_text s)) (B.neg (B.of_string s)));
      prop "abs n is n's decimal without its sign" text (fun s ->
          equal bigint (B.of_string (abs_text s)) (B.abs (B.of_string s)));
      prop "succ n is add n one" any (fun n ->
          equal bigint (B.add n B.one) (B.succ n));
      prop "pred n is sub n one" any (fun n ->
          equal bigint (B.sub n B.one) (B.pred n));
      prop "add n0 n1 is n0 + n1 on ints whose sum is an int"
        (int_pair half_int) (fun (x, y) ->
          equal bigint (B.of_int (x + y)) (B.add (B.of_int x) (B.of_int y)));
      prop "add is commutative" (Gen.pair any any)
        (Law.commutative bigint B.add);
      prop "add is associative" (Gen.triple any any any)
        (Law.associative bigint B.add);
      prop "sub n0 n1 is n0 - n1 on ints whose difference is an int"
        (int_pair half_int) (fun (x, y) ->
          equal bigint (B.of_int (x - y)) (B.sub (B.of_int x) (B.of_int y)));
      prop "sub undoes add" (Gen.pair any any) (fun (a, b) ->
          equal bigint a (B.sub (B.add a b) b));
      prop "mul n0 n1 is n0 * n1 on ints whose product is an int"
        (int_pair narrow) (fun (x, y) ->
          equal bigint (B.of_int (x * y)) (B.mul (B.of_int x) (B.of_int y)));
      prop "mul is commutative" (Gen.pair any any)
        (Law.commutative bigint B.mul);
      prop "mul distributes over add" (Gen.triple any any any)
        (Law.distributive bigint B.mul ~over:B.add);
      test "mul (2^64 - 1) (2^64 - 1) is Python's (2**64 - 1)**2" (fun () ->
          let m = B.pred (power_of_two 64) in
          equal string "340282366920938463426481119284349108225"
            (B.to_string (B.mul m m)));
      prop "div a b is a / b rounded toward zero" ~examples:division_examples
        division (fun (a, b) -> rounds `Zero (B.div a b) (a, b));
      prop "rem a b is a - b * div a b" ~examples:division_examples division
        (fun (a, b) -> equal bigint (B.sub a (B.mul b (B.div a b))) (B.rem a b));
      prop "fdiv a b is a / b rounded toward negative infinity"
        ~examples:division_examples division (fun (a, b) ->
          rounds `Down (B.fdiv a b) (a, b));
      prop "cdiv a b is a / b rounded toward positive infinity"
        ~examples:division_examples division (fun (a, b) ->
          rounds `Up (B.cdiv a b) (a, b));
      cases "div, rem, fdiv and cdiv raise Division_by_zero if b is zero"
        ~id:fst
        [ ("div", B.div); ("rem", B.rem); ("fdiv", B.fdiv); ("cdiv", B.cdiv) ]
        (fun (_, f) ->
          List.iter
            (fun a -> raises Division_by_zero (fun () -> f a B.zero))
            [ B.zero; B.one; two_to 100 ]);
      prop
        "divisible a b is true iff a = b * q for an integer q, so only zero by \
         zero"
        ~examples:
          [
            (B.zero, B.zero);
            (B.one, B.zero);
            (two_to 100, B.zero);
            (int_min, B.minus_one);
          ]
        (Gen.with_pp pp_pair (Gen.frequency [ (1, pair); (1, division) ]))
        (fun (a, b) ->
          let expected =
            if B.sign b = 0 then B.sign a = 0
            else B.equal (B.mul b (B.div a b)) a
          in
          cover "zero by zero" (B.sign a = 0 && B.sign b = 0);
          cover "nonzero by zero" (B.sign a <> 0 && B.sign b = 0);
          cover "divisible by nonzero" (expected && B.sign b <> 0);
          cover "not divisible" (not expected);
          satisfies ~claim:"divisible iff a = b * q" pairs
            (fun (a, b) -> Bool.equal expected (B.divisible a b))
            (a, b));
      prop "gcd a b divides a and b" gcd_pair (fun (a, b) ->
          let g = B.gcd a b in
          satisfies ~claim:"divides a" bigint (fun g -> B.divisible a g) g;
          satisfies ~claim:"divides b" bigint (fun g -> B.divisible b g) g);
      prop "gcd is the greatest divisor: gcd (c * a) (c * b) is abs c * gcd a b"
        (Gen.triple any any any) (fun (a, b, c) ->
          equal bigint
            (B.mul (B.abs c) (B.gcd a b))
            (B.gcd (B.mul c a) (B.mul c b)));
      prop "gcd a b is never negative" gcd_pair (fun (a, b) ->
          at_least bigint ~than:B.zero (B.gcd a b));
      prop "gcd a zero is abs a" ~examples:[ int_min; B.zero ] any (fun a ->
          equal bigint (B.abs a) (B.gcd a B.zero));
      prop "pow b e is the product of e b's, one for e = 0"
        ~examples:[ (B.zero, 0); (B.minus_one, 7); (int_min, 2) ]
        (Gen.pair any (Gen.int_range 0 8))
        (fun (b, e) ->
          equal bigint
            (List.fold_left B.mul B.one (List.init e (fun _ -> b)))
            (B.pow b e));
      cases "pow raises Invalid_argument if e is negative" ~id:Int.to_string
        [ -1; min_int ] (fun e ->
          raises_match Exn.invalid_arg (fun () -> B.pow B.one e));
    ]

(* Bits *)

(* The [w] low bits of [n], then the bit repeated past them: 1 if [n] is
   negative. *)
let bits n w =
  List.init w (fun i -> B.to_int (B.extract n i 1))
  @ [ (if B.sign n < 0 then 1 else 0) ]

(* [f a b] has [op] of [a]'s and [b]'s bit at each bit. The width covers every
   bit where [a], [b] or the result differs from its sign. *)
let each_bit op f (a, b) =
  let r = f a b in
  let w = 1 + List.fold_left Int.max 0 (List.map B.numbits [ a; b; r ]) in
  cover "a negative operand" (B.sign a < 0 || B.sign b < 0);
  cover "an operand past int" (not (B.fits_int a && B.fits_int b));
  equal (list int) (List.map2 op (bits a w) (bits b w)) (bits r w)

let field_examples =
  [
    (B.minus_one, 0, Sys.int_size - 1);
    (B.minus_one, 0, Sys.int_size);
    (B.minus_one, 3, 100);
    (int_min, 0, 64);
    (int_min, Sys.int_size - 1, 1);
    (B.zero, 0, 1);
  ]

let bits_section =
  group "bits"
    [
      prop "logand n0 n1 is the and of each bit" (Gen.pair any any)
        (each_bit ( land ) B.logand);
      prop "logor n0 n1 is the or of each bit" (Gen.pair any any)
        (each_bit ( lor ) B.logor);
      prop "logxor n0 n1 is the exclusive or of each bit" (Gen.pair any any)
        (each_bit ( lxor ) B.logxor);
      prop "lognot n is -n - 1" any (fun n ->
          equal bigint (B.pred (B.neg n)) (B.lognot n));
      prop "shift_left n k is n * 2^k"
        ~examples:
          [
            (B.one, Sys.int_size - 1);
            (B.minus_one, Sys.int_size - 1);
            (int_max, 1);
            (B.zero, 1000);
          ]
        (Gen.pair any small_k)
        (fun (n, k) ->
          equal bigint (B.mul n (power_of_two k)) (B.shift_left n k));
      prop "shift_right n k is n / 2^k rounded toward negative infinity"
        ~examples:
          [ (B.minus_one, 1000); (int_min, Sys.int_size - 1); (int_min, 100) ]
        (Gen.pair any small_k)
        (fun (n, k) ->
          equal bigint (B.fdiv n (power_of_two k)) (B.shift_right n k));
      cases "shift_left and shift_right raise Invalid_argument if k is negative"
        ~id:fst
        [ ("shift_left", B.shift_left); ("shift_right", B.shift_right) ]
        (fun (_, f) ->
          raises_match Exn.invalid_arg (fun () -> f B.one (-1));
          raises_match Exn.invalid_arg (fun () -> f (two_to 100) (-1)));
      prop
        "extract n off len is the non-negative integer of bits off to off + \
         len - 1"
        ~examples:field_examples
        (Gen.triple any small_k (Gen.int_range 1 200))
        (fun (n, off, len) ->
          let m = power_of_two len in
          let shifted = B.fdiv n (power_of_two off) in
          equal bigint
            (B.sub shifted (B.mul m (B.fdiv shifted m)))
            (B.extract n off len));
      prop
        "signed_extract n off len is extract n off len read as a len-bit two's \
         complement integer"
        ~examples:field_examples
        (Gen.triple any small_k (Gen.int_range 1 200))
        (fun (n, off, len) ->
          let e = B.extract n off len in
          let negative = B.geq e (power_of_two (len - 1)) in
          cover "negative" negative;
          equal bigint
            (if negative then B.sub e (power_of_two len) else e)
            (B.signed_extract n off len));
      cases
        "extract and signed_extract raise Invalid_argument if off is negative \
         or len is not positive"
        ~id:(fun (name, _, off, len) ->
          Printf.sprintf "%s n %d %d" name off len)
        [
          ("extract", B.extract, -1, 3);
          ("extract", B.extract, 0, 0);
          ("extract", B.extract, 0, -1);
          ("signed_extract", B.signed_extract, -1, 3);
          ("signed_extract", B.signed_extract, 0, 0);
          ("signed_extract", B.signed_extract, 0, -1);
        ]
        (fun (_, f, off, len) ->
          raises_match Exn.invalid_arg (fun () -> f B.minus_one off len);
          raises_match Exn.invalid_arg (fun () -> f (two_to 100) off len));
      prop "numbits n is the k for which 2^(k-1) <= abs n < 2^k, 0 for zero"
        ~examples:[ B.zero; int_min; int_max ] any (fun n ->
          let k = B.numbits n in
          if B.sign n = 0 then equal int 0 k
          else (
            at_most bigint ~than:(B.abs n) (power_of_two (k - 1));
            less bigint ~than:(power_of_two k) (B.abs n)));
      prop
        "trailing_zeros n is the greatest k for which 2^k divides n, max_int \
         for zero"
        ~examples:[ B.zero; int_min; two_to 300 ]
        any
        (fun n ->
          let k = B.trailing_zeros n in
          if B.sign n = 0 then equal int max_int k
          else (
            satisfies ~claim:"2^k divides n" bigint
              (fun n -> B.divisible n (power_of_two k))
              n;
            satisfies ~claim:"2^(k+1) does not divide n" bigint
              (fun n -> not (B.divisible n (power_of_two (k + 1))))
              n));
      prop "popcount n is the number of one bits of n"
        ~examples:[ B.zero; B.pred (two_to 300); int_max ]
        (Gen.map B.abs any)
        (fun n ->
          let ones = List.fold_left ( + ) 0 (bits n (B.numbits n)) in
          equal int ones (B.popcount n));
      cases "popcount raises Overflow if n is negative" ~id:fst
        [
          ("-1", B.minus_one);
          ("min_int", int_min);
          ("-2^100", B.neg (two_to 100));
        ]
        (fun (_, n) -> raises B.Overflow (fun () -> B.popcount n));
    ]

(* Operators *)

let operators =
  group "operators"
    [
      prop "each operator is the value it names" (Gen.pair pair small_k)
        (fun ((a, b), k) ->
          equal bigint (B.add a b) B.(a + b);
          equal bigint (B.sub a b) B.(a - b);
          equal bigint (B.mul a b) B.(a * b);
          if B.sign b <> 0 then equal bigint (B.div a b) B.(a / b);
          equal bigint (B.shift_left a k) B.(a lsl k);
          satisfies ~claim:"= < <= > >= are equal, lt, leq, gt, geq" pairs
            (fun (a, b) ->
              Bool.equal B.(a = b) (B.equal a b)
              && Bool.equal B.(a < b) (B.lt a b)
              && Bool.equal B.(a <= b) (B.leq a b)
              && Bool.equal B.(a > b) (B.gt a b)
              && Bool.equal B.(a >= b) (B.geq a b))
            (a, b));
    ]

let () =
  exit
    (run "Tolk.Bigint"
       [
         integers;
         constants;
         conversions;
         order;
         arithmetic;
         bits_section;
         operators;
       ])
