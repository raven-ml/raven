(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Printing: the form nx.mli states, each float printed so that it reads back
   to itself, and every kind of value printed without raising. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let m = Nx_support.memory

module S2 = (val Nx.devices [ m 0; m 1 ])

let printed x = Nx.to_string x

let starts ~prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let has_prefix prefix s =
  if not (starts ~prefix s) then failf "%S does not start with %S" s prefix

(* Forms *)

let forms =
  [
    ( "a matrix",
      (fun () -> printed (Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])),
      "float32 [2; 3]\n[[1, 2, 3],\n [4, 5, 6]]" );
    ( "a 0-d value",
      (fun () -> printed (Nx.create Nx.int64 [||] [| -7L |])),
      "int64 [] -7" );
    ( "unsigned integers",
      (fun () -> printed (Nx.create Nx.uint32 [| 2 |] [| -1l; 3l |])),
      "uint32 [2] [4294967295, 3]" );
    ( "complex numbers",
      (fun () ->
        printed
          (Nx.create Nx.complex64 [| 2 |]
             [| Complex.one; { Complex.re = 0.5; im = -2. } |])),
      "complex64 [2] [1+0i, 0.5-2i]" );
    ( "floats in their shortest form",
      (fun () -> printed (Nx.create Nx.float16 [| 4 |] [| 0.1; nan; neg_infinity; -0. |])),
      "float16 [4] [0.1, nan, -inf, -0]" );
    ( "floats to 8 significant digits",
      (fun () ->
        printed
          (Nx.create Nx.float64 [| 6 |]
             [| 1. /. 3.; 0.1 +. 0.2; 123456789.; 2e-7 /. 3.; 1e300 /. 3.; 9.999999996 |])),
      "float64 [6]\n[0.33333333, 0.3, 1.2345679e+08, 6.6666667e-08, 3.3333333e+299, 10]" );
    ( "large floats rounded above the units in exponent form",
      (fun () ->
        printed
          (Nx.create Nx.float64 [| 4 |]
             [| 1732540800123.; 100000000.5; 12345678.4; 99999999.7 |])),
      "float64 [4] [1.7325408e+12, 1e+08, 12345678, 100000000]" );
    ( "integers of at most 8 digits whole",
      (fun () ->
        printed (Nx.create Nx.float16 [| 3 |] [| 65504.; 2048.; -0. |])),
      "float16 [3] [65504, 2048, -0]" );
    ( "a float32 power of two whole up to 8 digits",
      (fun () -> printed (Nx.create Nx.float32 [| 2 |] [| 0x1p+24; 0x1p+30 |])),
      "float32 [2] [16777216, 1.0737418e+09]" );
    ( "a float32 integer past its digits in exponent form",
      (fun () -> printed (Nx.create Nx.float32 [| 2 |] [| 100000008.; 100000000. |])),
      "float32 [2] [1.0000001e+08, 100000000]" );
    ( "a float32 of 8 digits whole",
      (fun () -> printed (Nx.create Nx.float32 [| 2 |] [| 1. /. 3.; 16777216. |])),
      "float32 [2] [0.33333334, 16777216]" );
    ( "complex components to 8 significant digits",
      (fun () ->
        printed
          (Nx.create Nx.complex128 [| 1 |] [| { Complex.re = 1. /. 3.; im = -2. /. 3. } |])),
      "complex128 [1] [0.33333333-0.66666667i]" );
    ( "booleans", (fun () -> printed (Nx.create Nx.bool [| 2 |] [| true; false |])),
      "bool [2] [true, false]" );
    ("an empty value", (fun () -> printed (Nx.create Nx.int8 [| 0 |] [||])), "int8 [0] []");
    ( "an empty inner axis",
      (fun () -> printed (Nx.create Nx.int8 [| 2; 0 |] [||])),
      "int8 [2; 0]\n[[],\n []]" );
    ( "1000 elements in full",
      (fun () ->
        let s = printed (Nx.place Nx.Host.on (Nx.arange Nx.int16 0 1000 1)) in
        string_of_int (List.length (String.split_on_char ',' s))),
      "1000" );
    ( "past 1000, the ends of a long axis",
      (fun () -> printed (Nx.place Nx.Host.on (Nx.arange Nx.int32 0 1001 1))),
      "int32 [1001] [0, 1, 2, ..., 998, 999, 1000]" );
    ( "past 1000, NaN and infinities at the ends",
      (fun () ->
        let x = Nx.place Nx.Host.on (Nx.arange Nx.float32 0 1001 1) in
        let x = Nx.to_array x in
        x.(0) <- Float.nan;
        x.(1) <- Float.infinity;
        x.(1000) <- Float.neg_infinity;
        printed (Nx.create Nx.float32 [| 1001 |] x)),
      "float32 [1001] [nan, inf, 2, ..., 998, 999, -inf]" );
    ( "a 0-d value is never cut",
      (fun () -> printed (Nx.scalar Nx.float64 (1. /. 7.))),
      "float64 [] 0.14285714" );
    ( "an empty value of a long axis",
      (fun () -> printed (Nx.zeros Nx.int8 [| 0; 2000 |])),
      "int8 [0; 2000]\n[]" );
    ("pp_shape", (fun () -> Format.asprintf "%a" Nx.pp_shape [| 2; 3 |]), "[2; 3]");
    ("pp_shape of 0-d", (fun () -> Format.asprintf "%a" Nx.pp_shape [||]), "[]");
    ("pp_dtype", (fun () -> Format.asprintf "%a" Nx.pp_dtype Nx.bfloat16), "bfloat16");
  ]

let form_cases =
  cases "forms"
    ~id:(fun (n, _, _) -> n)
    forms
    (fun (_, f, expected) -> equal string expected (f ()))

(* Layouts: a row per line from two axes on, a long row wrapped. *)

let ints s =
  let n = Array.fold_left ( * ) 1 s in
  Nx.reshape s (Nx.place Nx.Host.on (Nx.arange Nx.int32 0 n 1))

let test_layouts () =
  expect (printed (ints [| 2; 2; 3 |])) @@ __POS_OF__ {|
int32 [2; 2; 3]
[[[0, 1, 2],
  [3, 4, 5]],
 [[6, 7, 8],
  [9, 10, 11]]]
|};
  expect (printed (ints [| 2; 30 |])) @@ __POS_OF__ {|
int32 [2; 30]
[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  21, 22, 23, 24, 25, 26, 27, 28, 29],
 [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]
|};
  expect (printed (ints [| 40; 30 |])) @@ __POS_OF__ {|
int32 [40; 30]
[[0, 1, 2, ..., 27, 28, 29],
 [30, 31, 32, ..., 57, 58, 59],
 [60, 61, 62, ..., 87, 88, 89],
 ...,
 [1110, 1111, 1112, ..., 1137, 1138, 1139],
 [1140, 1141, 1142, ..., 1167, 1168, 1169],
 [1170, 1171, 1172, ..., 1197, 1198, 1199]]
|};
  (* Every rank past 1000: axes of 6 or fewer whole, longer ones cut. *)
  expect (printed (ints [| 2; 3; 200 |])) @@ __POS_OF__ {|
int32 [2; 3; 200]
[[[0, 1, 2, ..., 197, 198, 199],
  [200, 201, 202, ..., 397, 398, 399],
  [400, 401, 402, ..., 597, 598, 599]],
 [[600, 601, 602, ..., 797, 798, 799],
  [800, 801, 802, ..., 997, 998, 999],
  [1000, 1001, 1002, ..., 1197, 1198, 1199]]]
|};
  expect (printed (ints [| 1; 7; 1; 150 |])) @@ __POS_OF__ {|
int32 [1; 7; 1; 150]
[[[[0, 1, 2, ..., 147, 148, 149]],
  [[150, 151, 152, ..., 297, 298, 299]],
  [[300, 301, 302, ..., 447, 448, 449]],
  ...,
  [[600, 601, 602, ..., 747, 748, 749]],
  [[750, 751, 752, ..., 897, 898, 899]],
  [[900, 901, 902, ..., 1047, 1048, 1049]]]]
|};
  (* Empty rows count as elements, so a long axis of them is cut too. *)
  expect (printed (Nx.zeros Nx.int8 [| 1001; 0 |])) @@ __POS_OF__ {|
int8 [1001; 0]
[[],
 [],
 [],
 ...,
 [],
 [],
 []]
|};
  expect (printed (ints [| 30 |])) @@ __POS_OF__ {|
int32 [30]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 25, 26, 27, 28, 29]
|};
  expect (printed (Nx.zeros Nx.int8 [| 0; 3 |])) @@ __POS_OF__ {|
int8 [0; 3]
[]
|};
  expect (printed (ints [| 1; 1 |])) @@ __POS_OF__ {|
int32 [1; 1]
[[0]]
|};
  expect (printed (ints [| 1; 3 |])) @@ __POS_OF__ {|
int32 [1; 3]
[[0, 1, 2]]
|};
  expect (printed (ints [| 3; 1 |])) @@ __POS_OF__ {|
int32 [3; 1]
[[0],
 [1],
 [2]]
|}

let test_boxed () =
  let v = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let s = Nx.scalar Nx.float32 1.5 in
  let boxed =
    Format.asprintf "@[<v 2>params:@,w = %a@,b = %a@,m = %a@]" Nx.pp v Nx.pp s
      Nx.pp m
  in
  expect boxed @@ __POS_OF__ {|
params:
  w = float32 [6] [1, 2, 3, 4, 5, 6]
  b = float32 [] 1.5
  m = float32 [2; 2]
      [[1, 2],
       [3, 4]]
|};
  equal string "w = float32 [6] [1, 2, 3, 4, 5, 6]"
    (Format.asprintf "@[<hv 2>w =@ %a@]" Nx.pp v);
  (* A matrix's rows break inside a box that fits. *)
  expect (Format.asprintf "@[<hv 2>m =@ %a@]" Nx.pp m) @@ __POS_OF__ {|
m = float32 [2; 2]
    [[1, 2],
     [3, 4]]
|};
  expect (Format.asprintf "@[<hov 2>m =@ %a@]" Nx.pp m) @@ __POS_OF__ {|
m = float32 [2; 2]
    [[1, 2],
     [3, 4]]
|};
  (* Without room beside the header, the elements still fill one line; a 0-d
     value's stay beside it, past a margin. *)
  let pad = String.make 50 '.' in
  expect (Format.asprintf "@[<hov 2>%s %a@]" pad Nx.pp v) @@ __POS_OF__ {|
.................................................. float32 [6]
                                                   [1, 2, 3, 4, 5, 6]
|};
  let narrow ppf =
    Format.pp_set_margin ppf 10;
    Nx.pp ppf s
  in
  equal string "float32 [] 1.5" (Format.asprintf "%t" narrow)

(* Values of every kind *)

let test_on_a_set () =
  let x = Nx.place (S2.split ~axis:0) (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  has_prefix "float32 [2] on set " (printed x);
  equal bool true (String.ends_with ~suffix:"[1, 2]" (printed x))

let test_host_creation () =
  equal string "float32 [2; 3]\n[[1, 1, 1],\n [1, 1, 1]]"
    (printed (Nx.add (Nx.zeros Nx.float32 [| 2; 3 |]) (Nx.scalar Nx.float32 1.)))

let test_dead () =
  let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
  let h = Nx.donate x in
  equal string "float32 [2], donated" (printed x);
  ignore (Nx.add h (Nx.create Nx.float32 [| 2 |] [| 0.; 0. |]));
  equal string "float32 [2], donated to Nx.add" (printed h);
  let m = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let h = Nx.donate m in
  ignore (Nx.add h (Nx.zeros Nx.float32 [| 2; 2 |]));
  equal string "float32 [2; 2], donated" (printed m);
  equal string "float32 [2; 2], donated to Nx.add" (printed h)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let test_traced () =
  let x = Nx.create Nx.int8 [| 3 |] [| 1; 2; 3 |] in
  Nx.Prim.interpret ~name:"test.print" Values
    (fun i ~by op -> Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op)
    (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form x) Mark in
      equal string "int8 [3] traced by test.print" (printed t);
      let m = Nx.Prim.traced i (Nx.Prim.form (Nx.reshape [| 3; 1 |] x)) Mark in
      equal string "int8 [3; 1] traced by test.print" (printed m))

(* Floats to 8 significant digits *)

let digits = 8

(* Random bits: every value of the format, NaNs and infinities included. *)
let drawn (type s) (dt : (float, s) D.t) n : (float, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string (if data = "" then "\000" else data)))

(* Random bits mixed with decimals of at most 8 digits and with decimals of 9
   digits from 10{^ 7} up to 10{^ 17}, which float32 and float64 rarely draw
   from bits alone. *)
let mixed (type s) (dt : (float, s) D.t) n : (float, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let short =
    let+ m = int_range 1 99_999_999 and+ e = int_range (-40) 40 in
    D.of_float dt (float_of_string (Printf.sprintf "%de%d" m e))
  in
  (* Bits from bytes, which draw every exponent alike. *)
  let bits =
    let+ b = string_of ~size:(constant 8) char in
    if D.bytes dt 1 = 4 then Int32.float_of_bits (String.get_int32_le b 0)
    else Int64.float_of_bits (String.get_int64_le b 0)
  in
  (* Nine digits about the units, rounded at the units or above. A float32
     needs 9 digits mostly at the start of a decade, where its step is finer
     than the 8th digit. *)
  let large =
    let+ m = int_range 100_000_000 199_999_999 and+ e = int_range (-1) 8 in
    D.of_float dt (float_of_string (Printf.sprintf "%de%d" m e))
  in
  (* Integers of up to 8 digits, which a float32 past 2{^ 24} spaces wider
     than 1. *)
  let integer =
    let+ k = int_range 0 99_999_999 in
    D.of_float dt (Float.of_int k)
  in
  let+ xs = array ~size:(constant n) (one_of [ short; bits; large; integer ]) in
  Nx.create dt [| n |] xs

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* [needed dt v] is the fewest significant digits of a decimal that reads back
   to the finite [v] in [dt]. *)
let needed (type s) (dt : (float, s) D.t) v =
  let rec go n =
    if n >= 17 then 17
    else if same (D.of_float dt (float_of_string (Printf.sprintf "%.*e" (n - 1) v))) v
    then n
    else go (n + 1)
  in
  go 1

(* The significant digits of a printed decimal: those from its first nonzero
   digit to its last. *)
let significant text =
  let mantissa = List.hd (String.split_on_char 'e' text) in
  let ds = String.concat "" (String.split_on_char '.' mantissa) in
  let ds = if ds <> "" && ds.[0] = '-' then String.sub ds 1 (String.length ds - 1) else ds in
  let first = ref (-1) and last = ref (-1) in
  String.iteri
    (fun i c ->
      if c <> '0' then begin
        if !first < 0 then first := i;
        last := i
      end)
    ds;
  if !first < 0 then 0 else !last - !first + 1

let elements s =
  let open_ = String.index s '[' in
  let body = String.sub s (open_ + 1) (String.length s - open_ - 1) in
  let open_ = String.index body '[' in
  let body = String.sub body (open_ + 1) (String.length body - open_ - 2) in
  if body = "" then [] else List.map String.trim (String.split_on_char ',' body)

(* [lead v] is the exponent of the leading digit of the finite [v]. *)
let lead v = int_of_string (List.nth (String.split_on_char 'e' (Printf.sprintf "%.16e" v)) 1)

(* Each printed element has at most 8 significant digits. It reads back to the
   element where a decimal of 8 digits does, and is otherwise within half a
   unit of its 8th digit, in exponent form where that unit is above 1. *)
let law_digits (type s) (dt : (float, s) D.t) x =
  let texts = Array.of_list (elements (printed x)) in
  let xs = Nx.to_array x in
  equal int (Array.length xs) (Array.length texts);
  Array.iteri
    (fun i v ->
      let t = texts.(i) in
      if Float.is_nan v then equal string "nan" t
      else if Float.abs v = Float.infinity then
        equal string (if v > 0. then "inf" else "-inf") t
      else begin
        less ~msg:t int ~than:(digits + 1) (significant t);
        let p = float_of_string t in
        (* An integer of at most 8 digits prints whole. *)
        if Float.is_integer v && Float.abs v < 1e8 then
          equal ~msg:(Printf.sprintf "%h" v) string
            ((if Float.sign_bit v then "-" else "") ^ Printf.sprintf "%.0f" (Float.abs v))
            t;
        (* A text of neither point nor exponent shows no digit [v] lacks: it
           is [v] to the units. *)
        if not (String.contains t '.' || String.contains t 'e') then
          less ~msg:(Printf.sprintf "%h printed as %s" v t) Windtrap.float_exact
            ~than:(0.5 +. (Float.succ (Float.abs v) -. Float.abs v))
            (Float.abs (p -. v));
        if needed dt v <= digits then begin
          classify "whole" true;
          if not (same v (D.of_float dt p)) then failf "%h printed as %s" v t
        end
        else begin
          classify "rounded" true;
          let e = lead v in
          let unit = Float.pow 10. (Float.of_int (e - digits + 1)) in
          if e >= digits then begin
            classify "rounded above the units" true;
            if not (String.contains t 'e') then
              failf "%h printed positionally as %s" v t
          end;
          (* Reading the decimal back to a double adds up to half a step of
             doubles, a whole one in the bound's own rounding. *)
          let step = Float.succ (Float.abs v) -. Float.abs v in
          less ~msg:(Printf.sprintf "%h printed as %s" v t) Windtrap.float_exact
            ~than:((0.5 *. unit *. (1. +. 1e-9)) +. step)
            (Float.abs (p -. v))
        end
      end)
    xs

let covered (type s) (dt : (float, s) D.t) x =
  let xs = Nx.to_array x in
  let finite = Array.to_list xs |> List.filter Float.is_finite in
  cover "an element rounded" (List.exists (fun v -> needed dt v > digits) finite);
  cover "an element whole" (List.exists (fun v -> needed dt v <= digits) finite);
  let rounded within = List.exists (fun v -> needed dt v > digits && within (lead v)) finite in
  (* A float32 from 10{^7} to 10{^8} is an integer of 8 digits. *)
  if D.bytes dt 1 = 8 then
    cover "an element rounded at the units" (rounded (fun e -> e = digits - 1));
  cover "an element rounded above the units, positional unrounded"
    (rounded (fun e -> e >= digits && e < 16));
  (* A float32 past 2{^ 24} is an integer whose shortest form may stop before
     its units. *)
  if D.bytes dt 1 = 4 then begin
    cover "an integer of at most 8 digits longer than its shortest form"
      (List.exists
         (fun v ->
           Float.is_integer v && Float.abs v < 1e8
           && needed dt v < String.length (Printf.sprintf "%.0f" (Float.abs v))
           && Float.rem (Float.abs v) 10. <> 0.)
         finite);
    cover "an integer of more digits than its shortest form"
      (List.exists
         (fun v ->
           let e = lead v in
           e < 16 && needed dt v <= e
           && Float.rem (Float.abs v) (Float.pow 10. (Float.of_int (e + 1 - needed dt v)))
              <> 0.)
         finite)
  end

let rounds =
  let floats =
    List.filter_map
      (fun (D.Any dt) ->
        match D.kind dt with
        | D.Float ->
            let long = D.bytes dt 1 >= 4 in
            let gen = if long then mixed dt 50 else drawn dt 50 in
            Some
              (prop
                 (D.name dt ^ " prints to 8 significant digits")
                 (Gen.with_pp (fun ppf _ -> D.pp ppf dt) gen)
                 (fun x ->
                   if long then covered dt x;
                   law_digits dt x))
        | _ -> None)
      D.all
  in
  group "each printed float has at most 8 significant digits" floats

(* [print]'s output on the standard formatter, read back through its out
   functions. *)
let test_print () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.5; -0.; Float.nan |] in
  let buffer = Buffer.create 64 in
  let saved = Format.pp_get_formatter_out_functions Format.std_formatter () in
  Format.pp_set_formatter_out_functions Format.std_formatter
    {
      saved with
      out_string = (fun s pos len -> Buffer.add_substring buffer s pos len);
      out_flush = ignore;
    };
  Fun.protect
    ~finally:(fun () ->
      Format.pp_set_formatter_out_functions Format.std_formatter saved)
    (fun () -> Nx.print x);
  equal ~msg:"print" string (Nx.to_string x ^ "\n") (Buffer.contents buffer)

let () =
  exit
    (run "nx print"
       [
         form_cases;
         group "values of every kind"
           [
             test "a row per line from two axes on" test_layouts;
             test "inside a box" test_boxed;
             test "a value on a set names the set" test_on_a_set;
             test "a host creation prints as a host value" test_host_creation;
             test "a dead value names its consumer" test_dead;
             test "a traced value names its interpretation" test_traced;
             test "print is to_string and a newline on stdout" test_print;
           ];
         rounds;
       ])
