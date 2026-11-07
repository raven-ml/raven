(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The decimal digits Dtype.pp_value and nx.io print floats with. The golden
   holds Python's and numpy's answers, checked against an exact search, for
   binary64 and binary32; the laws hold on every value of the narrow formats.
   The laws decide whether a decimal reads back with Float_reader, which
   shares no code with the module: [float_of_string] calls C's [strtod],
   which need not round correctly. *)

open Windtrap
module F = Nx_array.Dtype.Float_text
module R = Float_reader

let strf = Printf.sprintf
let digits = pair string int
let text (d, e) = strf "%se%d" d e
let single x = Int32.float_of_bits (Int32.bits_of_float x)
let reader (f : F.format) = { R.precision = f.precision; emin = f.emin }
let reads f x de = R.reads (reader f) x de

(* The float32s next to [x] above and below, for a float32 [x >= 0]. *)
let next32 x = Int32.float_of_bits (Int32.succ (Int32.bits_of_float x))
let prev32 x = Int32.float_of_bits (Int32.pred (Int32.bits_of_float x))

(* Formats *)

(* A narrow format: [exp] exponent bits and [frac] fraction bits, and [finite]
   its finite codes from zero up, which come first. *)
type narrow = { name : string; exp : int; frac : int; finite : int }

let narrows =
  [
    { name = "float16"; exp = 5; frac = 10; finite = 31 lsl 10 };
    { name = "bfloat16"; exp = 8; frac = 7; finite = 255 lsl 7 };
    { name = "float8_e4m3fn"; exp = 4; frac = 3; finite = 127 };
    { name = "float8_e5m2"; exp = 5; frac = 2; finite = 31 lsl 2 };
    { name = "float4_e2m1fn"; exp = 2; frac = 1; finite = 8 };
  ]

let format n = { F.precision = n.frac + 1; emin = 2 - (1 lsl (n.exp - 1)) }

(* The value of the code [c] of [n]. *)
let value n c =
  let emin = (format n).emin in
  let e = c lsr n.frac and m = c land ((1 lsl n.frac) - 1) in
  if e = 0 then Float.ldexp (Float.of_int m) (emin - n.frac)
  else Float.ldexp (Float.of_int (m lor (1 lsl n.frac))) (e - 1 + emin - n.frac)

(* Laws *)

(* [shorter (d, e)] are the decimals of one digit fewer next to [d 10^e] on
   either side: [m 10^(e+1)] and [(m+1) 10^(e+1)], with [m] the digits of [d]
   but the last. Since [d] ends in a nonzero digit, they lie on either side of
   it, and every decimal of fewer digits than [d] is at or beyond one of them.
   The reals that read as a float form an interval, so if neither reads as
   [x], no decimal shorter than [d] does. *)
let shorter (d, e) =
  let n = String.length d in
  if n < 2 then []
  else
    let m = int_of_string (String.sub d 0 (n - 1)) in
    [ (string_of_int m, e + 1); (string_of_int (m + 1), e + 1) ]

(* Of the shortest decimals, the nearest. [d 10^e] has [n] digits; the only
   others of [n] digits that can be nearer to [x] are its neighbours on the far
   side of [x]: [(d + 1) 10^e] above, and below [(d - 1) 10^e], or [9 10^(e-1)]
   when [d] is 1. One is nearer when [x] is past the midpoint toward it,
   [(10d + 5) 10^(e-1)] above and [(10d - 5) 10^(e-1)] or [95 10^(e-2)] below.
   At the midpoint the two tie, and the even last digit wins. [None] when the
   law holds. *)
let neighbours (d, e) =
  let m = int_of_string d in
  let up = ((string_of_int (m + 1), e), (string_of_int ((10 * m) + 5), e - 1))
  and down =
    if m = 1 then (("9", e - 1), ("95", e - 2))
    else ((string_of_int (m - 1), e), (string_of_int ((10 * m) - 5), e - 1))
  in
  [ (1, up); (-1, down) ]

let nearer f x (d, e) =
  let even = Char.code d.[String.length d - 1] land 1 = 0 in
  let beats (k, (n, mid)) =
    let o = R.compare mid x in
    x <> 0. && reads f x n && not (k * o > 0 || (o = 0 && even))
  in
  Option.map (fun (_, (n, _)) -> n) (List.find_opt beats (neighbours (d, e)))

(* [check f x] is why [shortest f x] breaks a law, or [None]. *)
let check f x =
  let de = F.shortest f x in
  if not (reads f x de) then Some (strf "%h: %s does not read back" x (text de))
  else
    match List.find_opt (reads f x) (shorter de) with
    | Some s -> Some (strf "%h: %s, but %s reads back" x (text de) (text s))
    | None -> (
        match nearer f x de with
        | Some n -> Some (strf "%h: %s, but %s is nearer" x (text de) (text n))
        | None -> None)

(* Inputs *)

let bits lo hi = Gen.map Int64.float_of_bits (Gen.int64_range lo hi)
let signed gen = Gen.(map (fun (x, s) -> if s then -.x else x) (pair gen bool))

(* [k] units in the last place from [x], positive and finite. *)
let ulps x k = Int64.float_of_bits (Int64.add (Int64.bits_of_float x) k)

(* [n 2^-r] with [n] of [p] bits and [r] from 1 to 3: its interval is 2^-r
   wide and its value ends in [5] at the digit after the grid of its shortest
   candidates, so two of them are often equally near. *)
let ties p =
  let open Gen in
  let+ n = int_range (1 lsl (p - 1)) ((1 lsl p) - 1) and+ r = int_range 1 3 in
  Float.ldexp (Float.of_int n) (-r)

let double =
  let open Gen in
  signed
    (frequency
       [
         (4, map Float.abs float);
         (1, bits 1L 0xF_FFFF_FFFF_FFFFL);
         ( 2,
           let+ e = int_range (-1022) 1023 and+ k = int_range (-2) 2 in
           ulps (Float.ldexp 1. e) (Int64.of_int k) );
         ( 2,
           let+ d = int_range 1 999 and+ j = int_range (-323) 305 in
           float_of_string (strf "%de%d" d j) );
         (1, map (fun k -> ulps max_float (Int64.of_int (-k))) (int_range 0 3));
         (2, ties 53);
       ])

let float32 =
  let open Gen in
  signed
    (frequency
       [
         (4, bits 0x3800_0000_0000_0000L 0x47EF_FFFF_E000_0000L);
         ( 1,
           map
             (fun k -> Float.ldexp (Float.of_int k) (-149))
             (int_range 1 0x7F_FFFF) );
         ( 2,
           let+ e = int_range (-126) 127 and+ k = int_range 0 2 in
           let x = Float.ldexp 1. e in
           if k = 0 then x else if k = 1 then next32 x else prev32 x );
         ( 2,
           let+ d = int_range 1 999 and+ j = int_range (-44) 35 in
           single (float_of_string (strf "%de%d" d j)) );
         (2, ties 24);
       ])
  |> Gen.map single

let classes32 x =
  let x = Float.abs x in
  cover "subnormal" (x > 0. && x < 0x1p-126);
  cover "binade bottom" (x >= 0x1p-126 && fst (Float.frexp x) = 0.5)

let classes64 x =
  let x = Float.abs x in
  cover "subnormal" (x > 0. && x < 0x1p-1022);
  cover "binade bottom" (x >= 0x1p-1022 && fst (Float.frexp x) = 0.5);
  cover "largest" (x > 0x1p1023)

let law_shortest name f gen classes =
  prop name ~count:2000 gen (fun x ->
      classes x;
      equal (option string) None (check f x))

(* Every finite value of a narrow format, of either sign. *)
let test_every_value n =
  let f = format n in
  let wrong = ref [] in
  for c = n.finite - 1 downto 0 do
    List.iter
      (fun x ->
        match check f x with Some w -> wrong := w :: !wrong | None -> ())
      [ value n c; -.value n c ]
  done;
  equal
    ~msg:(strf "%d values break a law" (List.length !wrong))
    (list string) []
    (List.filteri (fun i _ -> i < 8) !wrong)

(* A decimal of at most 15 digits in the normal range is the shortest of the
   double nearest it; of at most 6 digits, of the float32 nearest it. The
   nearest is found from the C library's reading, which is at most a unit in
   the last place off, by the exact reader. *)
let nearest f ~up ~down de guess =
  let near = [ guess; up guess; down guess ] in
  match List.find_opt (fun x -> reads f x de) near with
  | Some x -> x
  | None -> failf "%s reads as no value next to %h" (text de) guess

let short_decimal ~digits:p ~low ~high =
  let open Gen in
  let rec pow10 k = if k = 0 then 1 else 10 * pow10 (k - 1) in
  let+ m = int_range 1 (pow10 p - 1) and+ j = int_range low high in
  let s = string_of_int m in
  let k = ref (String.length s) in
  while s.[!k - 1] = '0' do
    decr k
  done;
  (String.sub s 0 !k, j + String.length s - !k)

let test_round_trip64 =
  prop "binary64 short decimal round trip" ~count:1000
    (short_decimal ~digits:15 ~low:(-300) ~high:290)
    (fun de ->
      let x =
        nearest F.binary64 ~up:Float.succ ~down:Float.pred de
          (float_of_string (text de))
      in
      equal digits de (F.shortest F.binary64 x))

let test_round_trip32 =
  prop "binary32 short decimal round trip" ~count:1000
    (short_decimal ~digits:6 ~low:(-36) ~high:30)
    (fun de ->
      let x =
        nearest F.binary32 ~up:next32 ~down:prev32 de
          (single (float_of_string (text de)))
      in
      equal digits de (F.shortest F.binary32 x))

(* [decimals n x] is right iff it is a multiple [m] of [10^-n] in canonical
   form, and [|x|] lies between the midpoints [(10m -+ 5) 10^(-n-1)] to its
   neighbours, on one only when [m] is even. *)
let canonical (d, e) =
  d = "0" && e = 0
  || (d <> "" && d.[0] <> '0' && d.[String.length d - 1] <> '0')

let pred_digits m =
  let b = Bytes.of_string m in
  let i = ref (Bytes.length b - 1) in
  while Bytes.get b !i = '0' do
    Bytes.set b !i '9';
    decr i
  done;
  Bytes.set b !i (Char.chr (Char.code (Bytes.get b !i) - 1));
  let s = Bytes.to_string b in
  if String.length s > 1 && s.[0] = '0' then
    String.sub s 1 (String.length s - 1)
  else s

let decimals_hold n x (d, e) =
  canonical (d, e)
  && (d = "0" || e >= -n)
  &&
  let m = if d = "0" then "0" else d ^ String.make (e + n) '0' in
  let even = Char.code m.[String.length m - 1] land 1 = 0 in
  let side o = o < 0 || (o = 0 && even) in
  (m = "0" || side (R.compare (pred_digits m ^ "5", -n - 1) x))
  && side (-R.compare (m ^ "5", -n - 1) x)

(* The exponent of the leading digit of the finite, nonzero [x]: [10^l <= |x|
   < 10^(l+1)], settled exactly from the float's estimate. *)
let lead x =
  let l = ref (Float.to_int (Float.floor (Float.log10 (Float.abs x)))) in
  while R.compare ("1", !l + 1) x <= 0 do
    incr l
  done;
  while R.compare ("1", !l) x > 0 do
    decr l
  done;
  !l

let finite = Gen.one_of [ double; float32 ]

(* Places past 40, which reach the end of long expansions. *)
let places = Gen.int_range 41 1100

let test_decimals =
  prop "decimals rounds to nearest, ties to even" ~count:2000
    Gen.(pair (frequency [ (9, int_range (-30) 40); (1, places) ]) finite)
    (fun (n, x) ->
      let de = F.decimals n x in
      cover "zero" (de = ("0", 0));
      equal
        ~msg:(strf "%d %h: %s" n x (text de))
        bool true (decimals_hold n x de))

(* [significant n x] rounds at the position of its [n]th digit, so it is
   [decimals (n - 1 - lead x) x], and has at most [n] digits. *)
let test_significant =
  prop "significant is decimals at the nth digit" ~count:2000
    Gen.(pair (int_range 1 40) finite)
    (fun (n, x) ->
      let de = F.significant n x in
      if x = 0. then equal digits ("0", 0) de
      else begin
        equal digits (F.decimals (n - 1 - lead x) x) de;
        equal ~msg:(text de) bool true (String.length (fst de) <= n)
      end)

(* The golden *)

(* The golden of gen/float_digits.py, by section: under a line naming it, the
   input's bits in hexadecimal, then [n] for significant and decimals, then the
   digits and the exponent of the answer. *)
type row = { hex : string; n : int; want : string * int }

let golden =
  let ic = open_in "golden/float_digits.txt" in
  let rec read acc =
    match In_channel.input_line ic with
    | None -> List.rev_map (fun (name, rows) -> (name, List.rev rows)) acc
    | Some line when String.starts_with ~prefix:"#" line -> read acc
    | Some line -> (
        match (String.split_on_char ' ' line, acc) with
        | ([ "shortest"; _ ] | [ ("significant" | "decimals") ]), _ ->
            read ((line, []) :: acc)
        | [ hex; d; e ], (name, rows) :: acc ->
            let row = { hex; n = 0; want = (d, int_of_string e) } in
            read ((name, row :: rows) :: acc)
        | [ hex; n; d; e ], (name, rows) :: acc ->
            let n = int_of_string n and want = (d, int_of_string e) in
            read ((name, { hex; n; want } :: rows) :: acc)
        | _ -> failf "golden: bad line %S" line)
  in
  Fun.protect ~finally:(fun () -> close_in ic) (fun () -> read [])

let of_bits64 hex = Int64.float_of_bits (Int64.of_string ("0x" ^ hex))
let of_bits32 hex = Int32.float_of_bits (Int32.of_string ("0x" ^ hex))

(* The function a section checks, from a row to its answer. *)
let section name =
  match String.split_on_char ' ' name with
  | [ "shortest"; "binary64" ] ->
      fun r -> F.shortest F.binary64 (of_bits64 r.hex)
  | [ "shortest"; "binary32" ] ->
      fun r -> F.shortest F.binary32 (of_bits32 r.hex)
  | [ "significant" ] -> fun r -> F.significant r.n (of_bits64 r.hex)
  | [ "decimals" ] -> fun r -> F.decimals r.n (of_bits64 r.hex)
  | _ -> failf "golden: unknown section %S" name

let test_golden (name, rows) =
  let answer = section name in
  let wrong =
    List.filter_map
      (fun r ->
        let got = answer r in
        if got = r.want then None
        else Some (strf "%s %d: %s, not %s" r.hex r.n (text got) (text r.want)))
      rows
  in
  equal
    ~msg:
      (strf "%s: %d of %d rows differ" name (List.length wrong)
         (List.length rows))
    (list string) []
    (List.filteri (fun i _ -> i < 8) wrong)

(* Stated cases *)

(* 5^1074, the digits of 2^-1074 = 5^1074 10^-1074, by schoolbook
   multiplication in base 10. *)
let pow5_1074 =
  let a = Array.make 800 0 in
  a.(0) <- 1;
  for _ = 1 to 1074 do
    let carry = ref 0 in
    for i = 0 to Array.length a - 1 do
      let t = (a.(i) * 5) + !carry in
      a.(i) <- t mod 10;
      carry := t / 10
    done
  done;
  let top = ref (Array.length a - 1) in
  while a.(!top) = 0 do
    decr top
  done;
  String.init (!top + 1) (fun i -> Char.chr (48 + a.(!top - i)))

let narrow name = format (List.find (fun n -> n.name = name) narrows)

let stated =
  [
    ("shortest zero", (fun () -> F.shortest F.binary64 0.), ("0", 0));
    ("shortest -0", (fun () -> F.shortest F.binary64 (-0.)), ("0", 0));
    ("shortest 0.1", (fun () -> F.shortest F.binary64 0.1), ("1", -1));
    ("shortest 5e-324", (fun () -> F.shortest F.binary64 5e-324), ("5", -324));
    ("shortest 1e23", (fun () -> F.shortest F.binary64 1e23), ("1", 23));
    ( "shortest 2^-1022",
      (fun () -> F.shortest F.binary64 0x1p-1022),
      ("22250738585072014", -324) );
    ( "shortest above 2^-1022",
      (fun () -> F.shortest F.binary64 (Float.succ 0x1p-1022)),
      ("2225073858507202", -323) );
    ( "shortest below 2^-1022",
      (fun () -> F.shortest F.binary64 (Float.pred 0x1p-1022)),
      ("2225073858507201", -323) );
    ( "shortest max_float",
      (fun () -> F.shortest F.binary64 max_float),
      ("17976931348623157", 292) );
    ( "shortest 2^50 + 0.25",
      (fun () -> F.shortest F.binary64 (0x1p50 +. 0.25)),
      ("11258999068426242", -1) );
    ( "shortest binary32 0.1",
      (fun () -> F.shortest F.binary32 (single 0.1)),
      ("1", -1) );
    ( "shortest float16 65504",
      (fun () -> F.shortest (narrow "float16") 65504.),
      ("655", 2) );
    (* The largest e4m3fn, 448, has an even significand: the reals from 440 to
       464 read as it, and 450 is the one of two digits. *)
    ( "shortest float8_e4m3fn 448",
      (fun () -> F.shortest (narrow "float8_e4m3fn") 448.),
      ("45", 1) );
    ( "shortest float8_e5m2 57344",
      (fun () -> F.shortest (narrow "float8_e5m2") 57344.),
      ("6", 4) );
    ( "shortest float4_e2m1fn 6",
      (fun () -> F.shortest (narrow "float4_e2m1fn") 6.),
      ("6", 0) );
    ( "significant 3 135.5625",
      (fun () -> F.significant 3 135.5625),
      ("136", 0) );
    ("significant 2 0.125", (fun () -> F.significant 2 0.125), ("12", -2));
    ( "significant max_int 0.005",
      (fun () -> F.significant max_int 0.005),
      ("5000000000000000104083408558608425664715468883514404296875", -60) );
    ( "significant max_int 5e-324",
      (fun () -> F.significant max_int 5e-324),
      (pow5_1074, -1074) );
    ("decimals 2 0.125", (fun () -> F.decimals 2 0.125), ("12", -2));
    ("decimals 0 0.5", (fun () -> F.decimals 0 0.5), ("0", 0));
    ("decimals 0 1.5", (fun () -> F.decimals 0 1.5), ("2", 0));
    ("decimals -2 1250", (fun () -> F.decimals (-2) 1250.), ("12", 2));
    ("decimals 2 -0.001", (fun () -> F.decimals 2 (-0.001)), ("0", 0));
    ("decimals max_int 1.5", (fun () -> F.decimals max_int 1.5), ("15", -1));
    ("decimals min_int 1.5", (fun () -> F.decimals min_int 1.5), ("0", 0));
  ]

let test_stated =
  cases "stated values" ~id:(fun (name, _, _) -> name) stated
    (fun (_, f, want) -> equal digits want (f ()))

(* The exact reader at ties to even and at the overflow threshold,
   1.797693134862315807...e308, and at e4m3fn's, 464. *)
let reader_rows =
  [
    ("0.1", F.binary64, 0.1, ("1", -1), true);
    ("2^53 + 1 to 2^53", F.binary64, 0x1p53, ("9007199254740993", 0), true);
    ( "2^53 + 1 to 2^53 + 2",
      F.binary64,
      0x1p53 +. 2.,
      ("9007199254740993", 0),
      false );
    ( "2^53 + 3 to 2^53 + 4",
      F.binary64,
      0x1p53 +. 4.,
      ("9007199254740995", 0),
      true );
    ( "2^53 + 3 to 2^53 + 2",
      F.binary64,
      0x1p53 +. 2.,
      ("9007199254740995", 0),
      false );
    ("below overflow", F.binary64, max_float, ("17976931348623158", 292), true);
    ("past overflow", F.binary64, max_float, ("17976931348623159", 292), false);
    ("e4m3fn 464", narrow "float8_e4m3fn", 448., ("464", 0), true);
    ("e4m3fn past 464", narrow "float8_e4m3fn", 448., ("4641", -1), false);
  ]

let test_reader =
  cases "exact reader" ~id:(fun (name, _, _, _, _) -> name) reader_rows
    (fun (_, f, x, de, want) -> equal bool want (reads f x de))

let refused =
  [
    ( "precision 64",
      fun () -> F.shortest { F.precision = 64; emin = -16382 } 0.1 );
    ("precision 0", fun () -> F.shortest { F.precision = 0; emin = 0 } 1.);
    ("binary32 0.1", fun () -> F.shortest F.binary32 0.1);
    ("e4m3fn 0.1", fun () -> F.shortest (narrow "float8_e4m3fn") 0.1);
    ("nan", fun () -> F.shortest F.binary64 Float.nan);
    ("infinity", fun () -> F.significant 3 Float.infinity);
    ("decimals of nan", fun () -> F.decimals 3 Float.nan);
    ("significant 0", fun () -> F.significant 0 1.);
  ]

let test_refused =
  cases "refused" ~id:fst refused (fun (_, f) -> raises_match Exn.invalid_arg f)

let () =
  exit
    (run "nx_array.float_text"
       [
         group "golden"
           [
             cases "answers as Python, numpy and an exact search give them"
               ~id:fst golden test_golden;
           ];
         group "laws"
           [
             law_shortest
               "binary64 shortest reads back, nothing shorter does, the nearest"
               F.binary64 double classes64;
             law_shortest
               "binary32 shortest reads back, nothing shorter does, the nearest"
               F.binary32 float32 classes32;
             cases
               "every value of a narrow format reads back, nothing shorter \
                does, the nearest"
               ~id:(fun n -> n.name)
               narrows test_every_value;
             test_round_trip64;
             test_round_trip32;
             test_decimals;
             test_significant;
           ];
         group "cases" [ test_stated; test_refused; test_reader ];
       ])
