open Windtrap
open Tolk
open Dtypes
module V = Dtype.Value

let rejects f = raises_match (Exn.invalid_arg ?sub:None) f
let with_setting setting name f = Setting.context [ B (setting, name) ] f

(* DEFAULT_FLOAT and DEFAULT_INT at their documented defaults, float32 and
   int32, whatever the environment sets: the goldens assume them. *)
let at_defaults f =
  Setting.context
    [ B (Setting.default_float, "float32"); B (Setting.default_int, "int32") ]
    f

let fmt_cell = function "None" -> None | s -> Some s.[0]
let python_bool b = if b then "True" else "False"
let dtype_list dts = Gen.of_list ~pp:(Testable.pp dtype) dts

let as_float = function
  | `Float f -> f
  | v -> failf "%a is not a float" (Testable.pp value) v

let as_int = function
  | `Int n -> n
  | v -> failf "%a is not an integer" (Testable.pp value) v

(* A golden cell [raises …] is a row where tinygrad raises, and where tolk
   rejects the argument. *)
let raised cell = String.starts_with ~prefix:"raises " cell

let is_cell w read cell f =
  if raised cell then rejects f else equal w (read cell) (f ())

(* Floats compared and printed by their bits, NaNs included. Only for values
   that are moved, never computed: every host keeps a moved NaN's bits. *)
let bits =
  Testable.make
    ~pp:(fun ppf x ->
      Format.fprintf ppf "%h (0x%016LX)" x (Int64.bits_of_float x))
    ~equal:(fun x y ->
      Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y))

let nan_of_bits = Int64.float_of_bits

(* Whether [x] is a NaN, and of which sign. *)
let nan_sign x =
  if not (Float.is_nan x) then "not a NaN"
  else if Float.sign_bit x then "-nan"
  else "+nan"

(* Exact values of floats *)

(* The exact value of a finite float [x], as [(m, e)] with [x = m * 2^e]. *)
let exact x =
  let f, e = Float.frexp x in
  (Bigint.of_float (Float.ldexp f 53), e - 53)

let align (m0, e0) (m1, e1) =
  let e = Int.min e0 e1 in
  (Bigint.shift_left m0 (e0 - e), Bigint.shift_left m1 (e1 - e))

let exact_compare a b =
  let m0, m1 = align a b in
  Bigint.compare m0 m1

let exact_sub a b =
  let m0, m1 = align a b in
  (Bigint.sub m0 m1, Int.min (snd a) (snd b))

(* Integers compared and ordered by value. *)
let ordered_z = Testable.with_compare Bigint.compare z
let exact_mul (m0, e0) (m1, e1) = (Bigint.mul m0 m1, e0 + e1)
let exact_abs (m, e) = (Bigint.abs m, e)

let exactly =
  Testable.with_compare exact_compare
    (Testable.make
       ~pp:(fun ppf (m, e) ->
         Format.fprintf ppf "%a * 2^%d" Bigint.pp_print m e)
       ~equal:(fun a b -> exact_compare a b = 0))

(* [nearest r x] is that the float [r] is a double nearest to the exact [x], the
   one with an even significand on a tie: no neighbour of [r] is nearer. *)
let nearest r x =
  let distance f = exact_abs (exact_sub (exact f) x) in
  let d = distance r in
  List.iter
    (fun n ->
      if Float.is_finite n then begin
        at_most exactly ~than:(distance n) d;
        if exact_compare (distance n) d = 0 then
          satisfies ~claim:"an even significand on a tie" float_exact
            (fun r -> Int64.equal (Int64.logand (Int64.bits_of_float r) 1L) 0L)
            r
      end)
    [ Float.pred r; Float.succ r ]

(* Constants *)

let nans =
  [
    Float.nan;
    Float.neg Float.nan;
    nan_of_bits 0x7FF0_0000_0000_0001L;
    nan_of_bits 0xFFF8_0000_DEAD_BEEFL;
  ]

let nan_id f = Printf.sprintf "0x%016LX" (Int64.bits_of_float f)

let any_const =
  Gen.with_pp (Testable.pp const)
    (Gen.frequency
       [
         (1, Gen.constant `Invalid);
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (3, Gen.map (fun n -> `Int n) integer);
         (2, Gen.map (fun f -> `Float f) Gen.any_float);
         (1, Gen.map (fun f -> `Float f) (Gen.of_list (0. :: -0. :: nans)));
       ])

(* The same constant built anew: an integer from its digits, and a float from
   its bits. *)
let respell = function
  | `Int n -> `Int (Bigint.of_string (Bigint.to_string n))
  | `Float f -> `Float (nan_of_bits (Int64.bits_of_float f))
  | c -> c

(* Pairs of constants, the second built anew from the first half the time: two
   drawn apart are rarely equal. *)
let const_pairs =
  Gen.(
    with_pp
      (fun ppf (a, b) ->
        Format.fprintf ppf "%a, %a" (Testable.pp const) a (Testable.pp const) b)
      (let* a = any_const in
       let+ b = frequency [ (1, constant (respell a)); (1, any_const) ] in
       (a, b)))

let const_equality =
  Testable.make ~pp:(Testable.pp const) ~equal:Dtype.equal_const

(* The same constructor with the same payload, floats by their bits. *)
let same_const c0 c1 =
  match (c0, c1) with
  | `Float f0, `Float f1 ->
      Int64.equal (Int64.bits_of_float f0) (Int64.bits_of_float f1)
  | _ -> Testable.equal const c0 c1

let nan_group =
  group "nan"
    [
      test "is the positive quiet NaN whose payload is zero" (fun () ->
          equal bits (nan_of_bits 0x7FF8_0000_0000_0000L) Dtype.nan);
    ]

let equal_const_group =
  group "equal_const"
    [
      prop
        "is true iff the constructor and payload are the same, floats by their \
         bits"
        ~examples:
          [
            (`Float 0., `Float (-0.));
            (`Int Bigint.one, `Float 1.);
            (`Bool true, `Int Bigint.one);
            (`Invalid, `Bool false);
          ]
        const_pairs
        (fun (c0, c1) ->
          let same = same_const c0 c1 in
          cover "the same constant" same;
          cover "different constants" (not same);
          if same then equal const_equality c0 c1
          else not_equal const_equality c0 c1);
      cases "tells apart NaNs of different bits" ~id:nan_id
        (List.filter
           (fun f ->
             not
               (Int64.equal (Int64.bits_of_float f)
                  (Int64.bits_of_float Float.nan)))
           nans)
        (fun f -> not_equal const_equality (`Float Float.nan) (`Float f));
    ]

let hash_const_group =
  group "hash_const"
    [
      prop "is the same for equal constants"
        ~examples:[ `Float Float.nan ]
        any_const
        (fun c ->
          cover "a NaN" (match c with `Float f -> Float.is_nan f | _ -> false);
          cover "an integer built anew"
            (match c with `Int _ -> true | _ -> false);
          equal int (Dtype.hash_const c) (Dtype.hash_const (respell c)));
    ]

let printed c = Format.asprintf "%a" Dtype.pp_const c

let pp_const_group =
  group "pp_const"
    [
      group "formats each constant as its golden states"
        [
          Golden.cases "const_repr.golden" (fun cell ->
              equal string (cell "printed")
                (printed (const_of_cell (cell "const"))));
        ];
      cases "formats every NaN as nan" ~id:nan_id nans (fun f ->
          equal string "nan" (printed (`Float f)));
      (* Bigint.to_string writes at most 4300 digits. *)
      test "formats an integer of 4300 digits" (fun () ->
          let n = Bigint.pred (Bigint.pow (Bigint.of_int 10) 4300) in
          equal string (String.make 4300 '9') (printed (`Int n)));
      test "raises on an integer of 4301 digits" (fun () ->
          rejects (fun () ->
              printed (`Int (Bigint.pow (Bigint.of_int 10) 4300))));
    ]

(* Values *)

(* An integer that rounds past the greatest double, from 2^1024 - 2^970, is the
   infinity of its sign where a float operates on it (Value's preamble). Python
   refuses to convert it, and its golden cell is a raise. *)
let beyond_doubles =
  Bigint.sub
    (Bigint.shift_left Bigint.one 1024)
    (Bigint.shift_left Bigint.one 970)

let to_infinity = function
  | `Int n when Bigint.geq (Bigint.abs n) beyond_doubles ->
      `Float (if Bigint.sign n < 0 then Float.neg_infinity else Float.infinity)
  | v -> v

(* [operates cell op a b] is that [op a b] is what [cell] states: a
   [Division_by_zero] where Python divides by zero, and where Python refuses a
   huge integer, [op] on the infinity of its sign. *)
let operates cell op a b =
  if String.equal cell "raises ZeroDivisionError" then
    raises Division_by_zero (fun () -> op a b)
  else if raised cell then
    match op (to_infinity a) (to_infinity b) with
    | expected -> equal value expected (op a b)
    | exception Division_by_zero -> raises Division_by_zero (fun () -> op a b)
  else equal value (value_of_cell cell) (op a b)

let magnitude =
  Testable.with_compare V.compare
    (Testable.make ~pp:(Testable.pp value) ~equal:(fun a b -> V.compare a b = 0))

let operand =
  Gen.with_pp (Testable.pp value)
    (Gen.frequency
       [
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (3, Gen.map (fun n -> `Int n) integer);
         (1, Gen.map (fun n -> `Int (Bigint.of_int n)) (Gen.int_range (-3) 3));
         (3, Gen.map (fun f -> `Float f) Gen.any_float);
         ( 1,
           Gen.map
             (fun f -> `Float f)
             (Gen.of_list [ 0.; -0.; 1.; -1.; 0x1p53 ]) );
       ])

let is_nan = function `Float f -> Float.is_nan f | _ -> false
let number = Gen.such_that (fun v -> not (is_nan v)) operand
let integers = Gen.map (fun n -> `Int n) integer

let is_zero = function
  | `Int n -> Bigint.equal n Bigint.zero
  | `Bool b -> not b
  | `Float f -> Stdlib.( = ) f 0.

let nonzero g = Gen.such_that (fun v -> not (is_zero v)) g

let pp_values ppf (a, b) =
  Format.fprintf ppf "%a, %a" (Testable.pp value) a (Testable.pp value) b

let value_pairs = Testable.make ~pp:pp_values ~equal:( = )
let pairs_of_values g = Gen.with_pp pp_values (Gen.pair g g)
let divisions g = Gen.with_pp pp_values (Gen.pair g (nonzero g))

(* An integer and a float, in either order. *)
let mixed =
  Gen.(
    with_pp pp_values
      (let+ n = map (fun n -> `Int n) integer
       and+ x = map (fun f -> `Float f) any_float
       and+ flip = bool in
       if flip then (x, n) else (n, x)))

let comparisons_with_nan v =
  let nan = `Float Float.nan in
  V.
    [
      ("nan = v", nan = v);
      ("v = nan", v = nan);
      ("nan < v", nan < v);
      ("v < nan", v < nan);
      ("nan <= v", nan <= v);
      ("v <= nan", v <= nan);
      ("nan > v", nan > v);
      ("nan >= v", nan >= v);
      ("nan <> v", nan <> v);
    ]

let preamble =
  [
    prop "counts a bool as 0 or 1" (Gen.pair Gen.bool operand) (fun (b, v) ->
        let n = `Int (if b then Bigint.one else Bigint.zero) in
        equal ~msg:"+" value V.(n + v) V.(`Bool b + v);
        equal ~msg:"*" value V.(n * v) V.(`Bool b * v);
        equal ~msg:"-" value V.(-n) V.(-`Bool b);
        if not (is_zero v) then begin
          equal ~msg:"//" value V.(n // v) V.(`Bool b // v);
          equal ~msg:"%" value V.(n % v) V.(`Bool b % v)
        end);
    prop "gives the exact integer of two integers" (pairs_of_values integers)
      (fun (a, b) ->
        let m = as_int a and n = as_int b in
        equal ~msg:"+" value (`Int (Bigint.add m n)) V.(a + b);
        equal ~msg:"-" value (`Int (Bigint.sub m n)) V.(a - b);
        equal ~msg:"*" value (`Int (Bigint.mul m n)) V.(a * b));
    prop "gives the double IEEE 754 gives of two floats"
      (Gen.pair Gen.any_float Gen.any_float) (fun (x, y) ->
        equal ~msg:"+" value (`Float (Float.add x y)) V.(`Float x + `Float y);
        equal ~msg:"-" value (`Float (Float.sub x y)) V.(`Float x - `Float y);
        equal ~msg:"*" value (`Float (Float.mul x y)) V.(`Float x * `Float y));
    (* 2^53 + 1 rounds to 2^53, and 2^53 + 0.5 to 2^53. Rounded once, 2^53 + 1.5
       would be 2^53 + 2. *)
    prop "rounds an integer to a double before it operates with a float"
      ~examples:[ (`Int (Bigint.of_string "9007199254740993"), `Float 0.5) ]
      mixed
      (fun (a, b) ->
        let double v = `Float (V.to_float v) in
        let a' = double a and b' = double b in
        equal ~msg:"+" value V.(a' + b') V.(a + b);
        equal ~msg:"-" value V.(a' - b') V.(a - b);
        equal ~msg:"*" value V.(a' * b') V.(a * b);
        if not (is_zero b) then begin
          equal ~msg:"//" value V.(a' // b') V.(a // b);
          equal ~msg:"%" value V.(a' % b') V.(a % b)
        end);
    prop "compares an integer and a float exactly"
      ~examples:
        [
          (Bigint.of_string "9007199254740993", 0x1p53);
          (Bigint.of_string "9007199254740992", 0x1p53);
        ]
      Gen.(
        let* n = integer in
        let+ x =
          frequency
            [
              (1, finite_float);
              (1, constant (Bigint.to_float n));
              (1, map (fun d -> Bigint.to_float n +. d) (float_range (-2.) 2.));
            ]
        in
        (n, x))
      (fun (n, x) ->
        assume (Float.is_finite x);
        let c = exact_compare (n, 0) (exact x) in
        cover "an integer that rounds to the float, unequal"
          (c <> 0 && Stdlib.( = ) (Bigint.to_float n) x);
        equal int (Int.compare c 0)
          (Int.compare (V.compare (`Int n) (`Float x)) 0));
    prop "makes every comparison with NaN false, but <>" operand (fun v ->
        equal
          (list (pair string bool))
          (List.map
             (fun (s, _) -> (s, String.equal s "nan <> v"))
             (comparisons_with_nan v))
          (comparisons_with_nan v));
    group "operates on each pair of values as its golden states"
      [
        Golden.cases "values.golden" ~key:[ "a"; "b" ] (fun cell ->
            let a = value_of_cell (cell "a") and b = value_of_cell (cell "b") in
            equal ~msg:"<" string (cell "lt") (python_bool V.(a < b));
            equal ~msg:"<=" string (cell "le") (python_bool V.(a <= b));
            equal ~msg:"=" string (cell "eq") (python_bool V.(a = b));
            equal ~msg:"<>" string (cell "ne") (python_bool V.(a <> b));
            equal ~msg:"min" value (value_of_cell (cell "min")) (V.min a b);
            equal ~msg:"max" value (value_of_cell (cell "max")) (V.max a b);
            operates (cell "add") V.( + ) a b;
            operates (cell "sub") V.( - ) a b;
            operates (cell "mul") V.( * ) a b;
            operates (cell "floordiv") V.( // ) a b;
            operates (cell "mod") V.( % ) a b);
      ];
    group "converts each value as its golden states"
      [
        Golden.cases "conversions.golden" (fun cell ->
            let v = value_of_cell (cell "value") in
            (match cell "float" with
            | c when raised c ->
                equal ~msg:"to_float" value (to_infinity v)
                  (`Float (V.to_float v))
            | c ->
                equal ~msg:"to_float" value (value_of_cell c)
                  (`Float (V.to_float v)));
            (match cell "int" with
            | c when raised c ->
                rejects (fun () -> V.to_z v);
                rejects (fun () -> V.to_int v)
            | c ->
                let n = Bigint.of_string c in
                equal ~msg:"to_z" z n (V.to_z v);
                if Bigint.fits_int n then
                  equal ~msg:"to_int" int (Bigint.to_int n) (V.to_int v)
                else rejects (fun () -> V.to_int v));
            equal ~msg:"to_bool" string (cell "bool")
              (python_bool (V.to_bool v)));
      ];
  ]

let of_int =
  group "of_int"
    [
      cases "is the integer" ~id:string_of_int
        [ Stdlib.min_int; -7; 0; Stdlib.max_int ] (fun n ->
          equal value (`Int (Bigint.of_int n)) (V.of_int n));
    ]

let to_float =
  group "to_float"
    [
      prop "is a float itself" Gen.any_float (fun f ->
          equal value (`Float f) (`Float (V.to_float (`Float f))));
      (* 2^53 + 1 and 2^53 + 3 lie halfway between two doubles. *)
      prop
        "is an integer's nearest double, ties to even, or the infinity of its \
         sign past the greatest double"
        ~examples:
          (List.map Bigint.of_string
             [ "9007199254740993"; "9007199254740995"; "-9007199254740993" ]
          @ [
              Bigint.pred beyond_doubles;
              beyond_doubles;
              Bigint.neg beyond_doubles;
            ])
        integer
        (fun n ->
          let r = V.to_float (`Int n) in
          cover "past the greatest double" (not (Float.is_finite r));
          if Float.is_finite r then nearest r (n, 0)
          else begin
            at_least ordered_z ~than:beyond_doubles (Bigint.abs n);
            equal int (Bigint.sign n) (Float.compare r 0.)
          end);
    ]

let to_z =
  group "to_z"
    [
      prop "rounds a float towards zero" finite_float (fun f ->
          equal z (Bigint.of_float (Float.trunc f)) (V.to_z (`Float f)));
      prop "is an integer itself" integer (fun n -> equal z n (V.to_z (`Int n)));
    ]

let to_int =
  group "to_int"
    [
      prop "is to_z where it fits an int, and raises otherwise"
        ~examples:
          Bigint.
            [
              of_int Stdlib.min_int;
              of_int Stdlib.max_int;
              pred (of_int Stdlib.min_int);
              succ (of_int Stdlib.max_int);
            ]
        integer
        (fun n ->
          if Bigint.fits_int n then
            equal int (Bigint.to_int n) (V.to_int (`Int n))
          else rejects (fun () -> V.to_int (`Int n)));
    ]

let to_bool =
  group "to_bool"
    [
      prop "is whether a value is not zero, a NaN true" operand (fun v ->
          satisfies ~claim:"to_bool v iff v is NaN or v <> 0" value
            (fun v -> Bool.equal (V.to_bool v) (is_nan v || V.(v <> of_int 0)))
            v);
    ]

let value_compare =
  group "compare"
    [
      prop "is a total order by magnitude"
        (Gen.triple operand operand operand)
        (Law.order magnitude);
      prop "agrees with < and = away from NaN" (pairs_of_values number)
        (fun (a, b) ->
          let c = V.compare a b in
          satisfies ~claim:"a < b iff compare a b < 0" value_pairs
            (fun (a, b) -> Bool.equal V.(a < b) (c < 0))
            (a, b);
          satisfies ~claim:"a = b iff compare a b = 0" value_pairs
            (fun (a, b) -> Bool.equal V.(a = b) (c = 0))
            (a, b));
      prop "puts NaN below every other value, and equal to itself" number
        (fun v ->
          let nan = `Float Float.nan in
          less int ~than:0 (V.compare nan v);
          equal int 0 (V.compare nan nan));
    ]

let derived =
  group "( <> ), ( <= ), ( > ) and ( >= )"
    [
      prop "are derived from < and =" (pairs_of_values operand) (fun (a, b) ->
          satisfies ~claim:"<> is not =" value_pairs
            (fun (a, b) -> Bool.equal V.(a <> b) (not V.(a = b)))
            (a, b);
          satisfies ~claim:"<= is < or =" value_pairs
            (fun (a, b) -> Bool.equal V.(a <= b) V.(a < b || a = b))
            (a, b);
          satisfies ~claim:"a > b is b < a" value_pairs
            (fun (a, b) -> Bool.equal V.(a > b) V.(b < a))
            (a, b);
          satisfies ~claim:"a >= b is b <= a" value_pairs
            (fun (a, b) -> Bool.equal V.(a >= b) V.(b <= a))
            (a, b));
    ]

(* Values of one magnitude and different kinds or signs, which min and max tell
   apart. *)
let ties =
  [
    (`Int Bigint.one, `Float 1.);
    (`Float 1., `Bool true);
    (`Float (-0.), `Float 0.);
    (`Float Float.nan, `Float 1.);
  ]

let tie (a, b) = V.(a = b) && not (Testable.equal value a b)

let min_group =
  group "min"
    [
      prop "is v1 if v1 < v0, and v0 otherwise" ~examples:ties
        (pairs_of_values operand) (fun (a, b) ->
          cover "a tie of different values" (tie (a, b));
          equal value (if V.(b < a) then b else a) (V.min a b));
    ]

let max_group =
  group "max"
    [
      prop "is v1 if v1 > v0, and v0 otherwise" ~examples:ties
        (pairs_of_values operand) (fun (a, b) ->
          cover "a tie of different values" (tie (a, b));
          equal value (if V.(b > a) then b else a) (V.max a b));
    ]

let negation =
  group "( ~- )"
    [
      group "negates each value as its golden states"
        [
          Golden.cases "negated.golden" (fun cell ->
              equal value
                (value_of_cell (cell "negated"))
                V.(-value_of_cell (cell "a")));
        ];
      prop "is an involution on integers and floats"
        (Gen.such_that (function `Bool _ -> false | _ -> true) operand)
        (Law.involutive value V.( ~- ));
    ]

let subtraction =
  group "( - )"
    [
      (* An integer zero has no sign, so -0.0 - 0 is -0.0 while -0.0 + -0 is
         0.0. *)
      prop "adds the negation of anything but an integer zero"
        (Gen.pair operand
           (Gen.such_that
              (function
                | `Int n -> not (Bigint.equal n Bigint.zero)
                | `Bool b -> b
                | `Float _ -> true)
              operand))
        (fun (a, b) -> equal value V.(a + -b) V.(a - b));
    ]

(* Floats to divide: any finite float, or a small integer, so that a remainder
   is often zero. *)
let dividend =
  Gen.frequency
    [ (3, finite_float); (1, Gen.map float_of_int (Gen.int_range (-20) 20)) ]

let pp_floats ppf (x, y) = Format.fprintf ppf "%h, %h" x y

let float_division =
  Gen.with_pp pp_floats
    (Gen.pair dividend (Gen.such_that (fun y -> Stdlib.( <> ) y 0.) dividend))

(* From 2^53 a quotient can round, and then it is not the exact floor. *)
let below_2_53 q = Float.abs q < 0x1p53

(* Each expected quotient is math.floor(Fraction(x) / Fraction(y)) in Python,
   converted to a float. *)
let floored =
  [
    (-8987075619625.855, 0.0028853025083442675, -3114777599102804.);
    (4562701585591296., -0.7670218673952995, -5948593879188349.);
    (-4.758569141785631e+17, -35.4099496915218, 1.3438508620431544e+16);
    (1e308, 1e-308, Float.infinity);
    (-1e308, 1e-308, Float.neg_infinity);
  ]

let floored_pairs = List.map (fun (x, y, _) -> (x, y)) floored

(* From first principles of a floor quotient [q] and its remainder [r], with [x
   = q * y + r] and [r] from 0 toward [y]: by an infinite [y], [x / y] is a zero
   of the exact quotient's sign, below 0 where the signs differ; an infinite [x]
   has no remainder, and a NaN is no number. *)
let non_finite =
  let inf = Float.infinity and nan = Float.nan in
  [
    (1., inf, 0., 1.);
    (-1., inf, -1., inf);
    (1., -.inf, -1., -.inf);
    (-1., -.inf, 0., -1.);
    (0., inf, 0., 0.);
    (-0., inf, -0., 0.);
    (0., -.inf, -0., -0.);
    (-0., -.inf, 0., -0.);
    (inf, 1., nan, nan);
    (-.inf, 2., nan, nan);
    (inf, inf, nan, nan);
    (nan, 1., nan, nan);
    (1., nan, nan, nan);
  ]

let float_quotient (x, y) = as_float V.(`Float x // `Float y)
let float_remainder (x, y) = as_float V.(`Float x % `Float y)

(* The exact remainder of [x] by [y] for the quotient [q]. *)
let exact_remainder x y q = exact_sub (exact x) (exact_mul (exact q) (exact y))
let zeros = [ `Int Bigint.zero; `Bool false; `Float 0.; `Float (-0.) ]
let value_id = Format.asprintf "%a" (Testable.pp value)

let floor_division =
  group "( // )"
    [
      prop "rounds an integer quotient down" (divisions integers) (fun (a, b) ->
          equal z (Bigint.fdiv (as_int a) (as_int b)) (as_int V.(a // b)));
      prop "is the floor of the exact quotient of floats, below 2^53"
        ~examples:
          (floored_pairs @ [ (1., 0.1); (5e-324, -4.450246113776542e-308) ])
        float_division
        (fun (x, y) ->
          let q = float_quotient (x, y) in
          cover "a quotient below 2^53" (below_2_53 q);
          cover "a negative quotient that is not whole"
            (q < 0. && not (Float.is_integer (x /. y)));
          if below_2_53 q then begin
            satisfies ~claim:"an integer" float_exact Float.is_integer q;
            (* q * y <= x < (q + 1) * y: the remainder is from 0 toward y. *)
            let r = exact_remainder x y q and zero = (Bigint.zero, 0) in
            if y > 0. then begin
              at_least exactly ~than:zero r;
              less exactly ~than:(exact y) r
            end
            else begin
              at_most exactly ~than:zero r;
              greater exactly ~than:(exact y) r
            end
          end);
      cases "is the floor of floats rounded once, an infinity past the doubles"
        ~id:(fun (x, y, _) -> Format.asprintf "%h // %h" x y)
        floored
        (fun (x, y, q) -> equal float_exact q (float_quotient (x, y)));
      prop "gives a zero quotient of floats the sign of v0 /. v1"
        ~examples:[ (0., -3.); (-0., 3.); (0.5, 1.); (-0., -1.) ]
        float_division
        (fun (x, y) ->
          let q = float_quotient (x, y) in
          cover "a zero quotient" (Stdlib.( = ) q 0.);
          if Stdlib.( = ) q 0. then equal bits (Float.copy_sign 0. (x /. y)) q);
      cases
        "divides by an infinity as its limit, and gives NaN of an infinity or \
         a NaN"
        ~id:(fun (x, y, _, _) -> Format.asprintf "%h // %h" x y)
        non_finite
        (fun (x, y, q, _) ->
          equal value (`Float q) (`Float (float_quotient (x, y))));
      cases "raises Division_by_zero if v1 is zero" ~id:value_id zeros
        (fun zero -> raises Division_by_zero (fun () -> V.(of_int 7 // zero)));
    ]

let remainder =
  group "( % )"
    [
      prop "rebuilds an integer dividend with its quotient" (divisions integers)
        (fun (a, b) -> equal value a V.((a // b * b) + (a % b)));
      prop "is an integer from 0 toward v1, v1 excluded" (divisions integers)
        (fun (a, b) ->
          let r = as_int V.(a % b) and n = as_int b in
          if Bigint.sign n > 0 then begin
            at_least ordered_z ~than:Bigint.zero r;
            less ordered_z ~than:n r
          end
          else begin
            at_most ordered_z ~than:Bigint.zero r;
            greater ordered_z ~than:n r
          end);
      (* Moving a remainder to the divisor's sign adds the divisor, which can
         round up to it: 5e-324 % -4.450246113776542e-308 is the divisor. *)
      prop
        "is the double nearest the exact remainder of floats, of v1's sign, \
         below 2^53"
        ~examples:
          (floored_pairs
          @ [
              (5e-324, -4.450246113776542e-308); (6., 3.); (-6., 3.); (6., -3.);
            ])
        float_division
        (fun (x, y) ->
          let q = float_quotient (x, y) and r = float_remainder (x, y) in
          cover "a remainder moved to v1's sign"
            (Stdlib.( <> ) r 0. && not (Bool.equal (x < 0.) (y < 0.)));
          cover "a zero remainder" (Stdlib.( = ) r 0.);
          equal ~msg:"sign" float_exact (Float.copy_sign 1. y)
            (Float.copy_sign 1. r);
          if below_2_53 q then nearest r (exact_remainder x y q));
      cases
        "divides by an infinity as its limit, and gives NaN of an infinity or \
         a NaN"
        ~id:(fun (x, y, _, _) -> Format.asprintf "%h %% %h" x y)
        non_finite
        (fun (x, y, _, r) ->
          equal value (`Float r) (`Float (float_remainder (x, y))));
      cases "raises Division_by_zero if v1 is zero" ~id:value_id zeros
        (fun zero -> raises Division_by_zero (fun () -> V.(`Float 1.5 % zero)));
    ]

let value_module =
  group "Value"
    (preamble
    @ [
        of_int;
        to_float;
        to_z;
        to_int;
        to_bool;
        value_compare;
        derived;
        min_group;
        max_group;
        negation;
        subtraction;
        floor_division;
        remainder;
      ])

let constants =
  group "Constants"
    [
      nan_group;
      equal_const_group;
      hash_const_group;
      pp_const_group;
      value_module;
    ]

(* Address spaces *)

let after prefix s =
  if not (String.starts_with ~prefix s) then
    failf "%S has no %s prefix" s prefix;
  String.sub s (String.length prefix) (String.length s - String.length prefix)

let space = Testable.make ~pp:Dtype.pp_addr_space ~equal:( = )

let address_spaces =
  group "Address spaces"
    [
      group "pp_addr_space"
        [
          cases "formats each address space as AddrSpace. and its name" ~id:snd
            Dtype.
              [
                (Global, "AddrSpace.GLOBAL");
                (Local, "AddrSpace.LOCAL");
                (Reg, "AddrSpace.REG");
                (Alu, "AddrSpace.ALU");
              ]
            (fun (space, printed) ->
              equal string printed
                (Format.asprintf "%a" Dtype.pp_addr_space space));
        ];
      group "addr_space_of_string"
        [
          cases "reads the name pp_addr_space formats after AddrSpace."
            ~id:(Format.asprintf "%a" Dtype.pp_addr_space)
            Dtype.[ Global; Local; Reg; Alu ]
            (Law.round_trip space string
               (Format.asprintf "%a" Dtype.pp_addr_space) (fun s ->
                 require_ok (Dtype.addr_space_of_string (after "AddrSpace." s))));
          cases "gives an error that names what is not an address space"
            ~id:(Printf.sprintf "%S")
            [ ""; "global"; "Global"; "AddrSpace.GLOBAL"; " GLOBAL"; "SHARED" ]
            (fun s ->
              contains ~sub:s (require_error (Dtype.addr_space_of_string s)));
        ];
    ]

(* Data types *)

let pairs_of dts = List.concat_map (fun a -> List.map (fun b -> (a, b)) dts) dts
let pair_id (a, b) = Printf.sprintf "%s, %s" (alias a) (alias b)

let properties =
  group "have the properties each data type's golden row states"
    [
      Golden.cases "properties.golden" (fun cell ->
          let dt = dtype_of_cell (cell "dtype") in
          equal ~msg:"pp" string (cell "dtype")
            (Format.asprintf "%a" Dtype.pp dt);
          equal ~msg:"priority" int
            (int_of_string (cell "priority"))
            (Dtype.priority dt);
          equal ~msg:"bitsize" int
            (int_of_string (cell "bitsize"))
            (Dtype.bitsize dt);
          equal ~msg:"itemsize" int
            (int_of_string (cell "itemsize"))
            (Dtype.itemsize dt);
          equal ~msg:"name" string (cell "name") (Dtype.name dt);
          equal ~msg:"fmt" (option char) (fmt_cell (cell "fmt")) (Dtype.fmt dt);
          equal ~msg:"min" value (value_of_cell (cell "min")) (Dtype.min dt);
          equal ~msg:"max" value (value_of_cell (cell "max")) (Dtype.max dt);
          equal ~msg:"is_int" string (cell "is_int")
            (python_bool (Dtype.is_int dt));
          equal ~msg:"is_float" string (cell "is_float")
            (python_bool (Dtype.is_float dt));
          equal ~msg:"is_unsigned" string (cell "is_unsigned")
            (python_bool (Dtype.is_unsigned dt));
          equal ~msg:"is_bool" string (cell "is_bool")
            (python_bool (Dtype.is_bool dt)));
    ]

(* The unsigned integer data type of [dt]'s words. *)
let word_of dt =
  match Dtype.itemsize dt with
  | 1 -> Dtype.Uint8
  | 2 -> Dtype.Uint16
  | 4 -> Dtype.Uint32
  | _ -> Dtype.Uint64

(* The word that stores [v] as a [dt] value. *)
let word dt v = as_int (Dtype.bitcast dt (word_of dt) v)

(* Where tolk departs from tinygrad, a value converts as tinygrad converts
   another, whose own golden row checks it.

   D9: a float rounds once to a narrow float. tinygrad's float32 step rounds
   1.0039062500000002 and 1 + 2^-8 + 2^-40 onto a bfloat16 tie, which then goes
   to even; rounded once, each is the bfloat16 above it.

   An integer from 2^1024 - 2^970 converts to a double as the infinity of its
   sign, and overflows a narrower float as the greatest double does (truncate),
   where Python refuses to convert it.

   D508: past an 8-bit float's greatest finite value, Fp8e5m2 gives the
   infinity of the value's sign, from 61440, where tinygrad saturates to
   57344, and Fp8e4m3 saturates an infinity to 448, where tinygrad gives a
   NaN. *)
let bfloat16_ties = [ 1.0039062500000002; 1. +. 0x1p-8 +. 0x1p-40 ]

let e5m2_overflow = 61440.

let as_tinygrad dt cell = function
  | `Float f when Dtype.equal dt Dtype.Bfloat16 && List.mem f bfloat16_ties ->
      Some (`Float 1.0078125)
  | `Float f when Dtype.equal dt Dtype.Fp8e4m3 && Float.abs f = Float.infinity
    ->
      Some (`Float (Float.copy_sign 448. f))
  | `Float f
    when Dtype.equal dt Dtype.Fp8e5m2
         && Float.is_finite f
         && Float.abs f >= e5m2_overflow ->
      Some (`Float (Float.copy_sign Float.infinity f))
  | `Int n
    when Dtype.equal dt Dtype.Fp8e5m2
         && Bigint.geq (Bigint.abs n) (Bigint.of_float e5m2_overflow)
         && not (raised cell) ->
      let s = if Bigint.sign n < 0 then -1. else 1. in
      Some (`Float (Float.copy_sign Float.infinity s))
  | `Int n
    when Dtype.is_float dt
         && Bigint.geq (Bigint.abs n) beyond_doubles
         && raised cell ->
      let big =
        if Dtype.(equal dt Float64 || equal dt Weak_float) then Float.infinity
        else Float.max_float
      in
      Some (`Float (if Bigint.sign n < 0 then -.big else big))
  | _ -> None

(* [row w read cell dt v f] checks [f v] against the golden row of [dt] and [v],
   or, where tolk departs from tinygrad, against [f] of the value that [v]
   converts as. *)
let row w read cell dt v f =
  match as_tinygrad dt cell v with
  | Some v' -> equal w (f v') (f v)
  | None -> is_cell w read cell (fun () -> f v)

(* D27: storage moves an Fp8e5m2 NaN's sign and the top two bits of its payload,
   quiet bit first, a quiet NaN's where those are zero, where tinygrad stores
   every NaN as S.11111.11. *)
let e5m2_nan_word x =
  let top = Int64.(to_int (shift_right_logical (bits_of_float x) 50)) land 3 in
  (if Float.sign_bit x then 0x80 else 0) lor 0x7C lor if top = 0 then 2 else top

let is_e5m2_nan dt = function
  | `Float f -> Dtype.equal dt Dtype.Fp8e5m2 && Float.is_nan f
  | _ -> false

(* A data type and a constant it takes: only a float or bool takes a NaN or an
   infinity, and only a float of known width is bounded. *)
let constable =
  Gen.bind every (fun dt ->
      let takes = function
        | `Float f -> Float.is_finite f || Dtype.is_float dt || Dtype.is_bool dt
        | _ -> true
      in
      Gen.map (fun c -> (dt, c)) (Gen.such_that takes any_const))

(* The words of the 8- or 16-bit float [dt] that fail [keeps]. *)
let words_failing dt keeps =
  let w = word_of dt in
  List.filter_map
    (fun word ->
      let word = `Int (Bigint.of_int word) in
      if keeps w word then None else Some word)
    (List.init (1 lsl (8 * Dtype.itemsize dt)) Fun.id)

let narrow_floats = Dtype.(Float16 :: Bfloat16 :: fp8s)

(* Signalling NaNs of 32 and 64 bits, as words. *)
let signalling =
  [
    (Dtype.Float32, `Int (Bigint.of_int 0x7F80_0001));
    (Dtype.Float32, `Int (Bigint.of_int 0xFFBF_FFFF));
    (Dtype.Float64, `Int (Bigint.of_string "0x7FF0000000000001"));
    (Dtype.Float64, `Int (Bigint.of_string "0xFFF7FFFFFFFFFFFF"));
  ]

let const_group =
  group "const"
    [
      group "is each constant of each data type as its golden states"
        [
          Golden.cases "const.golden" ~key:[ "dtype"; "value" ] (fun cell ->
              let dt = dtype_of_cell (cell "dtype")
              and c = const_of_cell (cell "value") in
              match c with
              | #Dtype.value as v ->
                  row const const_of_cell (cell "const") dt v (fun v ->
                      Dtype.const dt v)
              | `Invalid ->
                  is_cell const const_of_cell (cell "const") (fun () ->
                      Dtype.const dt c));
        ];
      prop "is idempotent" constable (fun (dt, c) ->
          Law.idempotent const (Dtype.const dt) c);
      cases "keeps the bits of every word of an 8- or 16-bit float" ~id:alias
        narrow_floats (fun dt ->
          equal (list value) []
            (words_failing dt (fun w word ->
                 match Dtype.const dt (Dtype.bitcast w dt word) with
                 | #Dtype.value as v ->
                     Testable.equal value word (Dtype.bitcast dt w v)
                 | `Invalid -> false)));
      prop "keeps the bits of a NaN of 32 or 64 bits, a signalling one included"
        ~examples:signalling
        (Gen.bind
           (dtype_list Dtype.[ Float32; Float64 ])
           (fun dt -> Gen.map (fun v -> (dt, v)) (value_of (word_of dt))))
        (fun (dt, word) ->
          let w = word_of dt in
          match Dtype.const dt (Dtype.bitcast w dt word) with
          | #Dtype.value as v -> equal value word (Dtype.bitcast dt w v)
          | `Invalid -> fail "Invalid");
    ]

let dtype_equality = Testable.make ~pp:(Testable.pp dtype) ~equal:Dtype.equal

let equal_group =
  group "equal"
    [
      cases "is true iff the data types are the same" ~id:pair_id
        (pairs_of declared) (fun (a, b) ->
          if a = b then equal dtype_equality a b
          else not_equal dtype_equality a b);
    ]

(* The promotion order's key: priority, then bitsize, then name. *)
let key dt = (Dtype.priority dt, Dtype.bitsize dt, Dtype.name dt)

let compare_group =
  group "compare"
    [
      cases "orders by priority, then bitsize, then name" ~id:pair_id
        (pairs_of declared) (fun (a, b) ->
          equal int
            (Int.compare (Stdlib.compare (key a) (key b)) 0)
            (Int.compare (Dtype.compare a b) 0));
      cases
        "orders Weak_int just after Bool, and each 8-bit float just before its \
         fnuz variant"
        ~id:pair_id
        Dtype.
          [ (Bool, Weak_int); (Fp8e4m3, Fp8e4m3fnuz); (Fp8e5m2, Fp8e5m2fnuz) ]
        (fun (a, b) ->
          equal (list dtype) [ b ]
            (List.filter
               (fun c -> Dtype.compare a c < 0 && Dtype.compare c b <= 0)
               declared));
    ]

let hash_group =
  group "hash"
    [
      cases "is the same for a data type read back from its name" ~id:alias
        declared (fun dt ->
          let read = require_ok (Dtype.of_string (Dtype.name dt)) in
          equal int (Dtype.hash dt) (Dtype.hash read));
    ]

let data_types =
  group "Data types"
    [ properties; const_group; equal_group; compare_group; hash_group ]

(* Predicates and groups *)

let predicates_and_groups =
  group "Predicates and groups"
    [
      group "are the lists their golden states"
        [
          Golden.cases "groups.golden" (fun cell ->
              let group =
                match cell "group" with
                | "fp8_ocp" -> Dtype.fp8_ocp
                | "fp8_fnuz" -> Dtype.fp8_fnuz
                | "fp8s" -> Dtype.fp8s
                | "floats" -> Dtype.floats
                | "uints" -> Dtype.uints
                | "sints" -> Dtype.sints
                | "ints" -> Dtype.ints
                | "weaks" -> Dtype.weaks
                | "all" -> Dtype.all
                | name -> invalid_arg name
              in
              equal (list dtype)
                (List.map dtype_of_cell
                   (String.split_on_char ' ' (cell "members")))
                group);
        ];
      group "mem"
        [
          prop "is list membership"
            (Gen.pair every (Gen.list every))
            (fun (dt, dts) ->
              cover "a member" (List.mem dt dts);
              cover "no member" (not (List.mem dt dts));
              satisfies ~claim:"mem dt dts iff List.mem dt dts"
                (pair dtype (list dtype))
                (fun (dt, dts) ->
                  Bool.equal (List.mem dt dts) (Dtype.mem dt dts))
                (dt, dts));
        ];
    ]

(* Literals *)

let literal =
  Gen.with_pp (Testable.pp const)
    (Gen.frequency
       [
         (1, Gen.constant `Invalid);
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         ( 2,
           Gen.map
             (fun n -> `Int (Bigint.of_int n))
             (Gen.int_range (-1000) 1000) );
         (2, Gen.map (fun n -> `Int n) integer);
         (2, Gen.map (fun f -> `Float f) Gen.any_float);
       ])

(* The outcome of [f ()]: its data type, or [None] where it raises. *)
let outcome f =
  match f () with dt -> Some dt | exception Invalid_argument _ -> None

(* The greatest of_const of [cs] by compare. *)
let greatest cs =
  List.fold_left
    (fun g c ->
      let dt = Dtype.of_const c in
      if Dtype.compare dt g > 0 then dt else g)
    Dtype.Bool cs

(* of_consts' definition, from the values it names. *)
let committed cs =
  if cs = [] then Dtype.default_float ()
  else if not (Dtype.equal (greatest cs) Dtype.Weak_int) then
    Dtype.strong (greatest cs)
  else
    let ints =
      List.filter_map
        (function
          | `Int n -> Some n
          | `Bool b -> Some (if b then Bigint.one else Bigint.zero)
          | _ -> None)
        cs
    in
    let lo = List.fold_left Bigint.min (List.hd ints) ints
    and hi = List.fold_left Bigint.max (List.hd ints) ints in
    Dtype.commit_int lo hi

let literals =
  group "Literals"
    [
      group "of_const"
        [
          group "is each literal's data type as its golden states"
            [
              Golden.cases "of_const.golden" (fun cell ->
                  equal dtype
                    (dtype_of_cell (cell "dtype"))
                    (Dtype.of_const (const_of_cell (cell "value"))));
            ];
        ];
      group "of_consts"
        [
          group "is each list's data type as its golden states"
            [
              Golden.cases "of_consts.golden" (fun cell ->
                  at_defaults (fun () ->
                      is_cell dtype dtype_of_cell (cell "dtype") (fun () ->
                          Dtype.of_consts (consts_of_cell (cell "consts")))));
            ];
          prop
            "commits the greatest literal's data type, an integer one to the \
             bounds of the integers"
            ~examples:
              [
                [];
                [ `Int (Bigint.shift_left Bigint.one 64) ];
                [ `Bool true; `Int (Bigint.of_string "-2147483649") ];
              ]
            (Gen.list literal)
            (fun cs ->
              cover "an integer list"
                (cs <> [] && Dtype.equal (greatest cs) Dtype.Weak_int);
              at_defaults (fun () ->
                  equal (option dtype)
                    (outcome (fun () -> committed cs))
                    (outcome (fun () -> Dtype.of_consts cs))));
        ];
      group "finfo"
        [
          group "is each float's widths as its golden states"
            [
              Golden.cases "finfo.golden" (fun cell ->
                  equal (pair int int)
                    ( int_of_string (cell "exponent"),
                      int_of_string (cell "mantissa") )
                    (Dtype.finfo (dtype_of_cell (cell "dtype"))));
            ];
          cases "raises outside floats" ~id:alias
            (List.filter (fun dt -> not (List.mem dt Dtype.floats)) declared)
            (fun dt -> rejects (fun () -> Dtype.finfo dt));
        ];
    ]

(* Names and defaults *)

(* A table of projections.golden, read at the default data types. *)
let projections claim check =
  group claim
    [
      Golden.cases "projections.golden" (fun cell ->
          at_defaults (fun () -> check cell (dtype_of_cell (cell "dtype"))));
    ]

let holds dt (lo, hi) =
  let min, max = int_bounds dt in
  Bigint.leq min lo && Bigint.leq hi max

let commit_bounds =
  Gen.map
    (fun (a, b) -> (Bigint.min a b, Bigint.max a b))
    (Gen.pair integer integer)

let pp_commit ppf (default_int, (lo, hi)) =
  Format.fprintf ppf "%a, %a, %a"
    (Format.pp_print_option
       ~none:(fun ppf () -> Format.pp_print_string ppf "None")
       (Testable.pp dtype))
    default_int Bigint.pp_print lo Bigint.pp_print hi

let u64_max = Bigint.of_string "18446744073709551615"

let names_and_defaults =
  group "Names and defaults"
    [
      group "of_string"
        [
          group "reads each name as its golden states"
            [
              Golden.cases "names.golden" (fun cell ->
                  at_defaults (fun () ->
                      equal (result dtype string)
                        (Ok (dtype_of_cell (cell "dtype")))
                        (Dtype.of_string (cell "name"))));
            ];
          cases "reads a name in any case" ~id:fst
            Dtype.
              [
                ("HALF", Float16);
                ("Float32", Float32);
                ("ULong", Uint64);
                ("BF16", Bfloat16);
              ]
            (fun (s, dt) ->
              equal (result dtype string) (Ok dt) (Dtype.of_string s));
          cases "reads the name pp formats after dtypes." ~id:alias declared
            (Law.round_trip dtype string (Format.asprintf "%a" Dtype.pp)
               (fun s -> require_ok (Dtype.of_string (after "dtypes." s))));
          cases "gives an error that names what is not a data type"
            ~id:(Printf.sprintf "%S")
            [
              "";
              "nonexistdtype";
              "f8";
              "i128";
              " half ";
              "uint128";
              "floats";
              "is_float";
              "dtypes";
              "dtypes.half";
              "dtypes.floats";
            ] (fun s -> contains ~sub:s (require_error (Dtype.of_string s)));
        ];
      group "default_float"
        [
          cases "is the float DEFAULT_FLOAT names" ~id:alias Dtype.floats
            (fun dt ->
              equal dtype dt
                (with_setting Setting.default_float (alias dt)
                   Dtype.default_float));
          test "reads DEFAULT_FLOAT in any case" (fun () ->
              equal dtype Dtype.Float16
                (with_setting Setting.default_float "HALF" Dtype.default_float));
          cases "raises on what is not a float of known width" ~id:Fun.id
            [ "int32"; "weakfloat"; "bool"; "void"; "typo" ] (fun name ->
              rejects (fun () ->
                  with_setting Setting.default_float name Dtype.default_float));
        ];
      group "default_int"
        [
          cases "is the integer DEFAULT_INT names" ~id:alias Dtype.ints
            (fun dt ->
              equal dtype dt
                (with_setting Setting.default_int (alias dt) Dtype.default_int));
          test "reads DEFAULT_INT in any case" (fun () ->
              equal dtype Dtype.Int64
                (with_setting Setting.default_int "LONG" Dtype.default_int));
          cases "raises on what is not an integer of known width" ~id:Fun.id
            [ "float32"; "weakint"; "bool"; "void"; "typo" ] (fun name ->
              rejects (fun () ->
                  with_setting Setting.default_int name Dtype.default_int));
        ];
      group "strong"
        [
          test "commits the weak data types at the current defaults" (fun () ->
              with_setting Setting.default_int "int64" (fun () ->
                  equal dtype Dtype.Int64 (Dtype.strong Dtype.Weak_int));
              with_setting Setting.default_float "float16" (fun () ->
                  equal dtype Dtype.Float16 (Dtype.strong Dtype.Weak_float)));
          projections "is each data type committed as its golden states"
            (fun cell dt ->
              equal dtype (dtype_of_cell (cell "strong")) (Dtype.strong dt));
        ];
      group "commit_int"
        [
          group "is each pair of bounds' data type as its golden states"
            [
              Golden.cases "commit.golden" ~key:[ "lo"; "hi"; "default_int" ]
                (fun cell ->
                  let lo = Bigint.of_string (cell "lo")
                  and hi = Bigint.of_string (cell "hi") in
                  let default_int =
                    match cell "default_int" with
                    | "None" -> None
                    | s -> Some (dtype_of_cell s)
                  in
                  at_defaults (fun () ->
                      is_cell dtype dtype_of_cell (cell "dtype") (fun () ->
                          Dtype.commit_int ?default_int lo hi)));
            ];
          prop
            "is the first of default_int, Int32, Int64 and Uint64 that holds \
             the bounds, else Int64"
            ~examples:
              [
                (Some Dtype.Int8, (Bigint.zero, Bigint.of_int 127));
                (None, (Bigint.zero, u64_max));
                (None, (Bigint.minus_one, u64_max));
                (None, (Bigint.succ u64_max, Bigint.succ u64_max));
              ]
            (Gen.with_pp pp_commit
               (Gen.pair (Gen.option (dtype_list Dtype.ints)) commit_bounds))
            (fun (default_int, (lo, hi)) ->
              at_defaults (fun () ->
                  let default = Option.value default_int ~default:Dtype.Int32 in
                  let first =
                    List.find_opt
                      (fun dt -> holds dt (lo, hi))
                      Dtype.[ default; Int32; Int64; Uint64 ]
                  in
                  cover "Uint64 alone holds them"
                    (first = Some Dtype.Uint64
                    && not (holds Dtype.Int64 (lo, hi)));
                  cover "none holds them" (first = None);
                  let expected =
                    match first with
                    | Some dt -> Some dt
                    | None when Bigint.equal lo hi -> None
                    | None -> Some Dtype.Int64
                  in
                  equal (option dtype) expected
                    (outcome (fun () -> Dtype.commit_int ?default_int lo hi))));
          cases "raises on a default_int outside ints" ~id:alias
            Dtype.[ Weak_int; Bool; Float32; Weak_float; Void ]
            (fun default_int ->
              rejects (fun () ->
                  Dtype.commit_int ~default_int Bigint.zero Bigint.one));
        ];
      group "weak"
        [
          projections "is each data type's weak data type as its golden states"
            (fun cell dt ->
              equal dtype (dtype_of_cell (cell "weak")) (Dtype.weak dt));
        ];
    ]

(* Promotion *)

let lub2 a b = Dtype.least_upper [ a; b ]
let promotes a b = Dtype.equal (lub2 a b) b

(* The least data type, by compare, that every one of [dts] promotes to. *)
let least_common_bound dts =
  List.filter
    (fun u -> List.for_all (fun d -> promotes d u) dts)
    (List.tl declared)
  |> List.sort Dtype.compare |> List.hd

let fold_bound dts = List.fold_left lub2 (List.hd dts) dts
let fp8_triple = Dtype.[ Fp8e4m3; Fp8e5m2; Bfloat16 ]

let lossless_cast =
  Gen.bind promotable (fun a ->
      Gen.bind
        (dtype_list
           (List.filter (Dtype.can_lossless_cast a) (List.tl declared)))
        (fun b -> Gen.map (fun v -> (a, b, v)) (value_of a)))

let promotion =
  group "Promotion"
    [
      group "least_upper"
        [
          group "is each pair's bound as its golden states"
            [
              Golden.cases "least_upper.golden" ~key:[ "a"; "b" ] (fun cell ->
                  equal dtype
                    (dtype_of_cell (cell "least_upper"))
                    (lub2 (dtype_of_cell (cell "a")) (dtype_of_cell (cell "b"))));
            ];
          group
            "is each triple's bound as its golden states, which its pairwise \
             fold is not always"
            [
              Golden.cases "least_upper_triples.golden" ~key:[ "a"; "b"; "c" ]
                (fun cell ->
                  let dts =
                    List.map (fun c -> dtype_of_cell (cell c)) [ "a"; "b"; "c" ]
                  in
                  equal ~msg:"least_upper" dtype
                    (dtype_of_cell (cell "least_upper"))
                    (Dtype.least_upper dts);
                  equal ~msg:"folded" dtype
                    (dtype_of_cell (cell "folded"))
                    (fold_bound dts));
            ];
          cases "is a data type itself" ~id:alias (List.tl declared) (fun dt ->
              equal dtype dt (Dtype.least_upper [ dt ]);
              equal dtype dt (Dtype.least_upper [ dt; dt ]));
          prop "orders the data types by promotion"
            ~examples:[ Dtype.(Bool, Int32, Float32) ]
            (Gen.map
               (fun (a, x, y) -> (a, lub2 a x, lub2 (lub2 a x) y))
               (Gen.triple promotable promotable promotable))
            (Law.partial_order dtype promotes);
          prop "is the least data type by compare that every one promotes to"
            ~examples:[ fp8_triple ]
            (Gen.list ~size:(Gen.int_range 1 5) promotable)
            (fun dts ->
              let bound = Dtype.least_upper dts in
              cover "a list whose pairwise fold is another bound"
                (not (Dtype.equal bound (fold_bound dts)));
              equal dtype (least_common_bound dts) bound);
          test "raises on the empty list" (fun () ->
              rejects (fun () -> Dtype.least_upper []));
          prop "raises on a list that holds Void" (Gen.list promotable)
            (fun dts ->
              rejects (fun () -> Dtype.least_upper (dts @ [ Dtype.Void ])));
        ];
      group "least_upper_float"
        [
          projections "is each data type's float as its golden states"
            (fun cell dt ->
              is_cell dtype dtype_of_cell (cell "least_upper_float") (fun () ->
                  Dtype.least_upper_float dt));
          cases "is a float itself, whatever the default" ~id:alias Dtype.floats
            (fun default ->
              with_setting Setting.default_float (alias default) (fun () ->
                  equal (list dtype) Dtype.floats
                    (List.map Dtype.least_upper_float Dtype.floats)));
          cases "is the default float of an integer" ~id:alias Dtype.floats
            (fun default ->
              with_setting Setting.default_float (alias default) (fun () ->
                  equal (list dtype)
                    (List.map (fun _ -> default) Dtype.ints)
                    (List.map Dtype.least_upper_float Dtype.ints)));
        ];
      group "can_lossless_cast"
        [
          group "is each pair's verdict as its golden states"
            [
              Golden.cases "lossless_cast.golden" ~key:[ "from"; "to" ]
                (fun cell ->
                  equal string (cell "lossless")
                    (python_bool
                       (Dtype.can_lossless_cast
                          (dtype_of_cell (cell "from"))
                          (dtype_of_cell (cell "to")))));
            ];
          prop "keeps every value of a cast it allows" lossless_cast
            (fun (a, b, v) ->
              Law.round_trip const const (Dtype.const b) (Dtype.const a)
                (v :> Dtype.const));
        ];
    ]

(* Casts *)

(* [nearest_bfloat16 x r], for a float [x >= 0], is that [r] is the bfloat16
   nearest to [x], the one with an even last bit on a tie, or an infinity from
   the midpoint past the largest bfloat16. A bfloat16's neighbours are one unit
   away in the upper half of its float32 bits, the largest one's upper neighbour
   is a unit of its binade above it, and the midpoint of two neighbours is exact
   in a float. *)
let nearest_bfloat16 x r =
  let bfloat16 top = Int32.float_of_bits (Int32.of_int (top lsl 16)) in
  let above top =
    if top = 0x7F7F then bfloat16 top +. (bfloat16 top -. bfloat16 (top - 1))
    else bfloat16 (top + 1)
  in
  let midpoint a b = (a +. b) /. 2. in
  if r = Float.infinity then
    at_least float_exact ~than:(midpoint (bfloat16 0x7F7F) (above 0x7F7F)) x
  else begin
    let bits = Int32.to_int (Int32.bits_of_float r) land 0xFFFF_FFFF in
    equal ~msg:"a bfloat16" int 0 (bits land 0xFFFF);
    let top = bits lsr 16 in
    let low = if top = 0 then 0. else midpoint (bfloat16 (top - 1)) r in
    let high = midpoint r (above top) in
    at_least float_exact ~than:low x;
    at_most float_exact ~than:high x;
    if x = low || x = high then equal ~msg:"even on a tie" int 0 (top land 1)
  end

let any_value =
  Gen.with_pp (Testable.pp value)
    (Gen.frequency
       [
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
         (2, Gen.map (fun n -> `Int n) integer);
         (3, Gen.map (fun f -> `Float f) Gen.any_float);
       ])

(* A data type and a value it truncates: an integer data type takes no float,
   and a float no integer beyond float64. *)
let truncatable =
  Gen.bind stored (fun dt ->
      let takes = function
        | `Float _ -> Dtype.is_float dt || Dtype.is_bool dt
        | `Int n -> (not (Dtype.is_float dt)) || Bigint.numbits n < 1000
        | `Bool _ -> true
      in
      Gen.map (fun v -> (dt, v)) (Gen.such_that takes any_value))

let truncated dt f = as_float (Dtype.truncate dt (`Float f))
let ones n = Bigint.pred (Bigint.shift_left Bigint.one n)

(* The word of a NaN of [dt], a float format with infinities, as the .mli states
   it: the sign of the double NaN [x], and as much of its payload as [dt] holds,
   quiet bit first, with the quiet bit set if [quiet] or if all that is zero. *)
let nan_word ~quiet dt x =
  let b = 8 * Dtype.itemsize dt and m = snd (Dtype.finfo dt) in
  let x = Bigint.extract (Bigint.of_int64 (Int64.bits_of_float x)) 0 64 in
  let negative = Bigint.extract x 63 1
  and payload = Bigint.extract x (52 - m) m in
  let payload =
    if quiet || Bigint.equal payload Bigint.zero then
      Bigint.logor payload (Bigint.shift_left Bigint.one (m - 1))
    else payload
  in
  let exponent = Bigint.shift_left (ones (b - 1 - m)) m in
  Bigint.(logor (shift_left negative (Int.sub b 1)) (logor exponent payload))

let with_infinities = Dtype.[ Fp8e5m2; Float16; Bfloat16; Float32; Float64 ]
let without_infinities = Dtype.[ Fp8e4m3; Fp8e4m3fnuz; Fp8e5m2fnuz ]

(* Quiet and signalling NaNs of both signs, with payloads at the top and at the
   bottom. *)
let nan_inputs =
  [
    0x7FF8_0000_0000_0000L;
    0xFFF8_0000_0000_0000L;
    0x7FF0_0000_0000_0001L;
    0x7FF4_0000_0000_0000L;
    0xFFF0_0000_0000_0001L;
    0x7FFF_FFFF_FFFF_FFFFL;
  ]

let nan_rows dts =
  List.concat_map (fun dt -> List.map (fun b -> (dt, b)) nan_inputs) dts

let nan_row_id (dt, b) = Printf.sprintf "%s 0x%016LX" (alias dt) b

(* The NaN an 8-bit float without infinities holds for one of sign
   [negative]. *)
let one_nan dt ~negative =
  if Dtype.equal dt Dtype.Fp8e4m3 && negative then "-nan" else "+nan"

let truncate_group =
  group "truncate"
    [
      group "rounds or wraps each value as its golden states"
        [
          Golden.cases "truncation.golden" ~key:[ "dtype"; "value" ]
            (fun cell ->
              let dt = dtype_of_cell (cell "dtype")
              and v = value_of_cell (cell "value") in
              row value value_of_cell (cell "truncated") dt v
                (Dtype.truncate dt));
        ];
      prop "rounds a float to Float32 as the host's IEEE conversion does"
        (Gen.frequency [ (1, Gen.any_float); (1, finite_float) ])
        (fun x ->
          (* On RISC-V a conversion to float32 makes every NaN its canonical
             one: value compares NaNs as a class. *)
          equal value
            (`Float (Int32.float_of_bits (Int32.bits_of_float x)))
            (Dtype.truncate Dtype.Float32 (`Float x)));
      prop "rounds once to Bfloat16, to the nearest, ties to even"
        ~examples:bfloat16_ties finite_float (fun f ->
          nearest_bfloat16 (Float.abs f)
            (truncated Dtype.Bfloat16 (Float.abs f)));
      prop "commutes with negation, but to an fnuz format's zero"
        (Gen.pair (dtype_list Dtype.floats) Gen.any_float)
        (fun (dt, f) ->
          let r = truncated dt f in
          let negated =
            if List.mem dt Dtype.fp8_fnuz && r = 0. then 0. else -.r
          in
          equal value (`Float negated) (Dtype.truncate dt (`Float (-.f))));
      prop
        "past an 8-bit float's greatest value is Fp8e5m2's infinity from 61440 \
         and Fp8e4m3's greatest value, an infinity included"
        ~examples:
          (List.concat_map
             (fun dt ->
               List.map
                 (fun f -> (dt, f))
                 [ Float.infinity; Float.neg_infinity; 61440.; -61440. ])
             Dtype.fp8s)
        (Gen.pair (dtype_list Dtype.fp8s) Gen.any_float)
        (fun (dt, f) ->
          let r = truncated dt f in
          let magnitude = Float.abs f in
          cover "an infinity" (magnitude = Float.infinity);
          cover "a finite value past 61440"
            (Float.is_finite f && magnitude >= e5m2_overflow);
          match dt with
          | _ when Float.is_nan f -> ()
          | Dtype.Fp8e5m2 ->
              equal bool ~msg:"an infinity" (magnitude >= e5m2_overflow)
                (Float.abs r = Float.infinity)
          | Fp8e4m3 ->
              satisfies ~claim:"finite" float_exact Float.is_finite r
          | _ ->
              equal bool ~msg:"a NaN" (magnitude = Float.infinity)
                (Float.is_nan r));
      cases
        "rounds an 8-bit float's greatest value, the tie past it and an infinity"
        ~id:(fun (dt, f, _) -> Format.asprintf "%a %h" Dtype.pp dt f)
        Dtype.
          [
            (Fp8e4m3, 448., 448.);
            (Fp8e4m3, 464., 448.);
            (Fp8e4m3, 480., 448.);
            (Fp8e4m3, -1e10, -448.);
            (Fp8e4m3, Float.infinity, 448.);
            (Fp8e4m3, Float.neg_infinity, -448.);
            (Fp8e5m2, 57344., 57344.);
            (Fp8e5m2, Float.pred 61440., 57344.);
            (Fp8e5m2, -.Float.pred 61440., -57344.);
            (Fp8e5m2, 61440., Float.infinity);
            (Fp8e5m2, -61440., Float.neg_infinity);
            (Fp8e5m2, 65536., Float.infinity);
            (Fp8e5m2, Float.infinity, Float.infinity);
            (Fp8e5m2, Float.neg_infinity, Float.neg_infinity);
          ]
        (fun (dt, f, want) -> equal float_exact want (truncated dt f));
      prop "is monotone on floats"
        (Gen.triple (dtype_list Dtype.floats) finite_float finite_float)
        (fun (dt, a, b) ->
          Law.monotone float_exact float_exact (truncated dt) (a, b));
      cases
        "quiets a NaN of a float with infinities, keeping its sign and the top \
         of its payload"
        ~id:nan_row_id (nan_rows with_infinities) (fun (dt, b) ->
          let x = nan_of_bits b in
          equal z
            (nan_word ~quiet:true dt x)
            (word dt (Dtype.truncate dt (`Float x))));
      cases "keeps the sign of an Fp8e4m3 NaN, and makes an fnuz NaN positive"
        ~id:nan_row_id (nan_rows without_infinities) (fun (dt, b) ->
          equal string
            (one_nan dt ~negative:(Int64.compare b 0L < 0))
            (nan_sign (truncated dt (nan_of_bits b))));
      (* Past 2^53 a double rounds an integer onto a tie of the narrower float,
         which it would then round to even: 9042383626829825 is 2^53 + 2^45 + 1,
         which a double makes the tie 2^53 + 2^45, rounded to 0x5A00, where it
         is 0x5A01 from the exact value. *)
      test "rounds an integer to a narrower float once, from its value"
        (fun () ->
          let z = Bigint.of_string "9042383626829825" in
          equal value
            (Dtype.bitcast Uint16 Bfloat16 (`Int (Bigint.of_int 0x5a01)))
            (Dtype.truncate Bfloat16 (`Int z));
          let z = Bigint.(shift_left one 60 + shift_left one 36 + one) in
          equal value
            (`Float (Float.ldexp 1. 60 +. Float.ldexp 1. 37))
            (Dtype.truncate Float32 (`Int z)));
      prop "wraps an integer modulo two to its width"
        (Gen.pair (dtype_list Dtype.ints) integer)
        (fun (dt, n) ->
          let r = as_int (Dtype.truncate dt (`Int n)) in
          let lo, hi = int_bounds dt in
          at_least ordered_z ~than:lo r;
          at_most ordered_z ~than:hi r;
          equal ~msg:"congruent" z Bigint.zero
            (Bigint.extract (Bigint.sub n r) 0 (Dtype.bitsize dt)));
      prop "is a value itself for a weak data type"
        (Gen.pair (dtype_list Dtype.weaks) any_value)
        (fun (dt, v) -> equal value v (Dtype.truncate dt v));
      prop "is idempotent" truncatable (fun (dt, v) ->
          Law.idempotent value (Dtype.truncate dt) v);
      test "raises on Void" (fun () ->
          rejects (fun () -> Dtype.truncate Dtype.Void (`Int Bigint.zero)));
    ]

let storage_fmt_group =
  group "storage_fmt"
    [
      projections "is each data type's storage format as its golden states"
        (fun cell dt ->
          equal (option char)
            (fmt_cell (cell "storage_fmt"))
            (Dtype.storage_fmt dt));
    ]

let to_storage_scalar_group =
  group "to_storage_scalar"
    [
      group "stores each value as its golden states"
        [
          Golden.cases "truncation.golden" ~key:[ "dtype"; "value" ]
            (fun cell ->
              let dt = dtype_of_cell (cell "dtype")
              and v = value_of_cell (cell "value") in
              if is_e5m2_nan dt v then
                equal value
                  (`Int (Bigint.of_int (e5m2_nan_word (as_float v))))
                  (Dtype.to_storage_scalar dt v)
              else
                row value value_of_cell (cell "storage") dt v
                  (Dtype.to_storage_scalar dt));
        ];
      cases
        "moves a NaN of a float with infinities, with its sign and as much of \
         its payload as it holds"
        ~id:nan_row_id (nan_rows with_infinities) (fun (dt, b) ->
          let x = nan_of_bits b in
          equal z (nan_word ~quiet:false dt x) (word dt (`Float x)));
      cases
        "stores an Fp8e4m3 NaN as 0x7F or 0xFF, of its sign, and an fnuz NaN \
         as 0x80"
        ~id:nan_row_id (nan_rows without_infinities) (fun (dt, b) ->
          let expected =
            if not (Dtype.equal dt Dtype.Fp8e4m3) then 0x80
            else if Int64.compare b 0L < 0 then 0xFF
            else 0x7F
          in
          equal value
            (`Int (Bigint.of_int expected))
            (Dtype.to_storage_scalar dt (`Float (nan_of_bits b))));
      prop "is undone by from_storage_scalar"
        (Gen.bind stored (fun dt -> Gen.map (fun v -> (dt, v)) (value_of dt)))
        (fun (dt, v) ->
          Law.round_trip value value
            (Dtype.to_storage_scalar dt)
            (Dtype.from_storage_scalar dt)
            v);
    ]

let word_id (dt, w) = Printf.sprintf "%s 0x%X" (alias dt) w

let decoded dt w =
  as_float (Dtype.from_storage_scalar dt (`Int (Bigint.of_int w)))

let from_storage_scalar_group =
  group "from_storage_scalar"
    [
      group "decodes each word as its golden states"
        [
          Golden.cases "decode.golden" ~key:[ "dtype"; "storage" ] (fun cell ->
              equal value
                (value_of_cell (cell "value"))
                (Dtype.from_storage_scalar
                   (dtype_of_cell (cell "dtype"))
                   (value_of_cell (cell "storage"))));
        ];
      cases "gives a Bfloat16 or Fp8e5m2 NaN the sign and payload of its bits"
        ~id:word_id
        Dtype.
          [
            (Bfloat16, 0x7FC0);
            (Bfloat16, 0xFFC0);
            (Bfloat16, 0x7F81);
            (Bfloat16, 0xFFFF);
            (Fp8e5m2, 0x7D);
            (Fp8e5m2, 0x7E);
            (Fp8e5m2, 0x7F);
            (Fp8e5m2, 0xFD);
            (Fp8e5m2, 0xFE);
          ]
        (fun (dt, w) ->
          let m = snd (Dtype.finfo dt) and b = 8 * Dtype.itemsize dt in
          let sign = Int64.of_int (w lsr (b - 1))
          and payload = Int64.of_int (w land ((1 lsl m) - 1)) in
          let expected =
            Int64.(
              logor (shift_left sign 63)
                (logor 0x7FF0_0000_0000_0000L (shift_left payload (52 - m))))
          in
          equal bits (nan_of_bits expected) (decoded dt w));
      cases
        "gives an Fp8e4m3 NaN the sign of its bits, and an fnuz NaN positive"
        ~id:word_id
        Dtype.
          [
            (Fp8e4m3, 0x7F);
            (Fp8e4m3, 0xFF);
            (Fp8e4m3fnuz, 0x80);
            (Fp8e5m2fnuz, 0x80);
          ]
        (fun (dt, w) ->
          equal string
            (one_nan dt ~negative:(w = 0xFF))
            (nan_sign (decoded dt w)));
      cases "raises on what is not an integer" ~id:alias
        Dtype.(Bfloat16 :: fp8s)
        (fun dt ->
          rejects (fun () -> Dtype.from_storage_scalar dt (`Float 1.));
          rejects (fun () -> Dtype.from_storage_scalar dt (`Bool true)));
    ]

(* Two data types a program stores, of one item size, and a value of the
   first. *)
let same_size =
  Gen.bind stored (fun a ->
      Gen.bind
        (dtype_list
           (List.filter
              (fun b ->
                Dtype.itemsize a = Dtype.itemsize b
                && not (List.mem b Dtype.[ Void; Weak_int; Weak_float ]))
              declared))
        (fun b -> Gen.map (fun v -> (a, b, v)) (value_of a)))

let pp_bitcast ppf (a, b, v) =
  Format.fprintf ppf "%a to %a of %a" Dtype.pp a Dtype.pp b (Testable.pp value)
    v

let bitcast_group =
  group "bitcast"
    [
      group "reads each value's bits as its golden states"
        [
          Golden.cases "bitcasts.golden" ~key:[ "from"; "to"; "value" ]
            (fun cell ->
              let a = dtype_of_cell (cell "from")
              and b = dtype_of_cell (cell "to") in
              let v = value_of_cell (cell "value") in
              if is_e5m2_nan a v then
                equal value
                  (Dtype.bitcast Dtype.Uint8 b
                     (`Int (Bigint.of_int (e5m2_nan_word (as_float v)))))
                  (Dtype.bitcast a b v)
              else
                row value value_of_cell (cell "bitcast") a v (Dtype.bitcast a b));
        ];
      (* A Bool keeps whether its byte is zero, which is nx's meaning of a
         boolean byte. *)
      prop
        "is undone by the bitcast back, but into a Bool of a byte neither 0 \
         nor 1"
        ~examples:
          ((Dtype.Uint8, Dtype.Bool, `Int (Bigint.of_int 2))
          :: (Dtype.Uint8, Dtype.Fp8e5m2, `Int (Bigint.of_int 0x7D))
          :: (Dtype.Uint8, Dtype.Fp8e4m3, `Int (Bigint.of_int 0xFF))
          :: List.map (fun (dt, w) -> (word_of dt, dt, w)) signalling)
        (Gen.with_pp pp_bitcast same_size)
        (fun (a, b, v) ->
          let back = Dtype.bitcast b a (Dtype.bitcast a b v) in
          let byte = word a v in
          let lossy =
            Dtype.equal b Dtype.Bool
            && not
                 (Bigint.equal byte Bigint.zero || Bigint.equal byte Bigint.one)
          in
          cover "into a Bool, a byte neither 0 nor 1" lossy;
          cover "through a float" (Dtype.is_float b);
          if lossy then
            equal value (Dtype.bitcast Dtype.Uint8 a (`Int Bigint.one)) back
          else equal value v back);
      cases "gives back every word through an 8- or 16-bit float" ~id:alias
        narrow_floats (fun dt ->
          equal (list value) []
            (words_failing dt (fun w word ->
                 Testable.equal value word
                   (Dtype.bitcast dt w (Dtype.bitcast w dt word)))));
      test "reads a byte as a Bool true iff the byte is not zero" (fun () ->
          let bytes = List.init 256 Fun.id in
          equal (list value)
            (List.map (fun b -> `Bool (b <> 0)) bytes)
            (List.map
               (fun b ->
                 Dtype.bitcast Dtype.Uint8 Dtype.Bool (`Int (Bigint.of_int b)))
               bytes));
      test "raises on data types of different item sizes" (fun () ->
          rejects (fun () ->
              Dtype.bitcast Dtype.Int8 Dtype.Float16 (`Int Bigint.one)));
      cases "raises on a data type with no storage" ~id:alias
        Dtype.[ Void; Weak_int; Weak_float ]
        (fun dt ->
          rejects (fun () -> Dtype.bitcast dt dt (`Int Bigint.one));
          rejects (fun () -> Dtype.bitcast Dtype.Int64 dt (`Int Bigint.one)));
    ]

let casts =
  group "Casts"
    [
      truncate_group;
      storage_fmt_group;
      to_storage_scalar_group;
      from_storage_scalar_group;
      bitcast_group;
    ]

let () =
  exit
    (run "Tolk.Dtype"
       [
         constants;
         address_spaces;
         data_types;
         predicates_and_groups;
         literals;
         names_and_defaults;
         promotion;
         casts;
       ])
